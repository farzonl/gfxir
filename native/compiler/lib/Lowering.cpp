#include "Gfx/Gfx.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <functional>

namespace mlir::gfx {
#define GEN_PASS_DEF_GFXTOSPIRV
#define GEN_PASS_DEF_GFXTOLLVM
#define GEN_PASS_DEF_GFXVERIFY
#include "Gfx/GfxPasses.h.inc"

namespace {
LogicalResult graphicsEntry(ModuleOp module, func::FuncOp &function,
                            Stage &stage) {
  if (failed(verifyShaderModule(module)))
    return failure();
  for (auto entry : module.getOps<func::FuncOp>())
    if (auto attribute = entry->getAttrOfType<StageAttr>("gfx.stage")) {
      function = entry;
      stage = attribute.getValue();
    }
  if (stage != Stage::Vertex && stage != Stage::Fragment &&
      stage != Stage::Compute)
    return function.emitOpError(
        "this graphics-first lowering supports vertex, fragment, and compute stages; "
        "task/mesh require additional target legalization");
  if (!function.getBody().hasOneBlock())
    return function.emitOpError(
        "this foundation supports single-block shaders; structured "
        "control-flow lowering is not implemented yet");
  return success();
}

LogicalResult
collectGraphicsFunctions(ModuleOp module, func::FuncOp entry,
                         SmallVectorImpl<func::FuncOp> &reachable) {
  llvm::SmallPtrSet<Operation *, 8> active, visited;
  std::function<LogicalResult(func::FuncOp)> visit =
      [&](func::FuncOp current) -> LogicalResult {
    if (active.contains(current))
      return current.emitOpError("recursive shader calls are unsupported");
    if (!visited.insert(current).second)
      return success();
    if (current.isExternal() || !current.getBody().hasOneBlock() ||
        current.getFunctionType().getNumResults() > 1)
      return current.emitOpError("graphics helpers require a single structured "
                                 "body and at most one result");
    active.insert(current);
    auto calls = current.walk([&](func::CallOp call) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee || (callee != entry && callee->hasAttr("gfx.stage"))) {
        call.emitOpError("requires a defined ordinary shader helper");
        return WalkResult::interrupt();
      }
      if (failed(visit(callee)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    active.erase(current);
    reachable.push_back(current);
    return calls.wasInterrupted() ? failure() : success();
  };
  return visit(entry);
}

struct GfxVerifyPass : impl::GfxVerifyBase<GfxVerifyPass> {
  void runOnOperation() override {
    if (failed(verifyShaderModule(getOperation())))
      signalPassFailure();
  }
};

struct GfxToSPIRVPass : impl::GfxToSPIRVBase<GfxToSPIRVPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp function;
    Stage stage;
    if (failed(graphicsEntry(module, function, stage))) {
      signalPassFailure();
      return;
    }
    DenseI32ArrayAttr wgSizeAttr =
        function->getAttrOfType<DenseI32ArrayAttr>("gfx.workgroup_size");
    auto *context = module.getContext();
    auto location = function.getLoc();
    SmallVector<func::FuncOp> reachable;
    if (failed(collectGraphicsFunctions(module, function, reachable))) {
      signalPassFailure();
      return;
    }
    auto entryName = function.getSymName().str();
    OpBuilder builder(context);
    // SPIR-V 1.0 requires a componentwise predicate for vector selects.
    module.walk([&](arith::SelectOp select) {
      auto vectorType = dyn_cast<VectorType>(select.getType());
      if (vectorType && select.getCondition().getType().isInteger(1)) {
        builder.setInsertionPoint(select);
        auto conditionType =
            VectorType::get(vectorType.getShape(), builder.getI1Type());
        select.getConditionMutable().assign(vector::BroadcastOp::create(
            builder, select.getLoc(), conditionType, select.getCondition()));
      }
    });
    builder.setInsertionPointToStart(module.getBody());
    auto version = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_0, {spirv::Capability::Shader}, {}, context);
    auto target = spirv::TargetEnvAttr::get(
        version, spirv::getDefaultResourceLimits(context),
        spirv::ClientAPI::Vulkan);
    auto shader = spirv::ModuleOp::create(builder, location,
                                          spirv::AddressingModel::Logical,
                                          spirv::MemoryModel::GLSL450, version);
    function->removeAttr("gfx.stage");
    function->removeAttr("gfx.workgroup_size");
    for (auto helper : reachable)
      helper->moveBefore(shader.getBody(), shader.getBody()->end());
    for (auto helper :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      helper.erase();
    SymbolTable symbols(shader);
    auto parameters =
        module->getAttrOfType<ParameterBlockAttr>("gfx.parameters");
    spirv::GlobalVariableOp parameterVariable;
    bool readsParameters = false;
    shader.walk([&](ParameterOp) { readsParameters = true; });
    if (readsParameters) {
      auto layout = parameterLayout(parameters);
      SmallVector<Type> types;
      SmallVector<uint32_t> offsets;
      for (const auto &field : layout.fields) {
        types.push_back(field.field.getType());
        offsets.push_back(field.offset);
      }
      auto structure = spirv::StructType::get(types, offsets);
      auto pointer =
          spirv::PointerType::get(structure, spirv::StorageClass::PushConstant);
      builder.setInsertionPoint(function);
      parameterVariable = spirv::GlobalVariableOp::create(
          builder, location, pointer, "parameters");
      symbols.insert(parameterVariable);
      SmallVector<ParameterOp> reads;
      shader.walk([&](ParameterOp op) { reads.push_back(op); });
      for (auto op : reads) {
        auto field = llvm::find_if(layout.fields, [&](const auto &field) {
          return field.field.getName() == op.getField();
        });
        builder.setInsertionPoint(op);
        auto address =
            spirv::AddressOfOp::create(builder, op.getLoc(), parameterVariable);
        auto index = spirv::ConstantOp::create(
            builder, op.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(field - layout.fields.begin()));
        auto member = spirv::AccessChainOp::create(builder, op.getLoc(),
                                                   address, ValueRange{index});
        auto value = spirv::LoadOp::create(builder, op.getLoc(), member);
        op.getValue().replaceAllUsesWith(value);
        op.erase();
      }
    }
    module->removeAttr("gfx.parameters");
    SmallVector<Attribute> interface;
    SmallVector<Operation *> operations;
    function.walk([&](Operation *op) {
      if (isa<BuiltinOp, InputOp, OutputOp, PositionOp>(op))
        operations.push_back(op);
    });
    unsigned nextSymbol = 0;
    for (Operation *op : operations) {
      bool input = isa<BuiltinOp, InputOp>(op);
      Type type =
          input ? op->getResult(0).getType() : op->getOperand(0).getType();
      bool fragmentCoord =
          isa<BuiltinOp>(op) &&
          cast<BuiltinOp>(op).getKind() == Builtin::FragmentCoord;
      bool globalInvocationId =
          isa<BuiltinOp>(op) &&
          cast<BuiltinOp>(op).getKind() == Builtin::GlobalInvocationId;
      if (fragmentCoord)
        type = VectorType::get({4}, builder.getF32Type());
      auto pointerType =
          spirv::PointerType::get(type, input ? spirv::StorageClass::Input
                                              : spirv::StorageClass::Output);
      builder.setInsertionPoint(function);
      auto variable = spirv::GlobalVariableOp::create(
          builder, op->getLoc(), pointerType,
          "interface_" + std::to_string(nextSymbol++));
      symbols.insert(variable);
      if (isa<BuiltinOp>(op))
        variable->setAttr(
            "built_in",
            builder.getStringAttr(
                fragmentCoord ? "FragCoord"
                              : (globalInvocationId ? "GlobalInvocationId"
                                                    : "VertexIndex")));
      else if (isa<PositionOp>(op))
        variable->setAttr("built_in", builder.getStringAttr("Position"));
      else {
        int32_t index = input ? cast<InputOp>(op).getLocation()
                              : cast<OutputOp>(op).getLocation();
        Interpolation mode = input ? cast<InputOp>(op).getInterpolation()
                                   : cast<OutputOp>(op).getInterpolation();
        variable->setAttr("location", builder.getI32IntegerAttr(index));
        if (mode == Interpolation::Flat)
          variable->setAttr("flat", builder.getUnitAttr());
        else if (mode == Interpolation::NoPerspective)
          variable->setAttr("no_perspective", builder.getUnitAttr());
      }
      interface.push_back(FlatSymbolRefAttr::get(variable.getSymNameAttr()));
      builder.setInsertionPoint(op);
      auto address =
          spirv::AddressOfOp::create(builder, op->getLoc(), variable);
      if (input) {
        Value value = spirv::LoadOp::create(builder, op->getLoc(), address);
        if (fragmentCoord) {
          SmallVector<Value> xy;
          for (int32_t index : {0, 1})
            xy.push_back(spirv::CompositeExtractOp::create(
                builder, op->getLoc(), value, ArrayRef<int32_t>{index}));
          value = spirv::CompositeConstructOp::create(
              builder, op->getLoc(), op->getResult(0).getType(), xy);
        }
        op->getResult(0).replaceAllUsesWith(value);
      } else {
        Value value = op->getOperand(0);
        if (isa<PositionOp>(op)) {
          auto y = spirv::CompositeExtractOp::create(
              builder, op->getLoc(), value, ArrayRef<int32_t>{1});
          auto inverted =
              spirv::FNegateOp::create(builder, op->getLoc(), y.getType(), y);
          value = spirv::CompositeInsertOp::create(
              builder, op->getLoc(), inverted, value, ArrayRef<int32_t>{1});
        }
        spirv::StoreOp::create(builder, op->getLoc(), address, value);
      }
      op->erase();
    }
    SPIRVTypeConverter converter(target);
    RewritePatternSet patterns(context);
    ScfToSPIRVContext scfContext;
    arith::populateArithToSPIRVPatterns(converter, patterns);
    populateMathToSPIRVPatterns(converter, patterns);
    populateVectorToSPIRVPatterns(converter, patterns);
    populateBuiltinFuncToSPIRVPatterns(converter, patterns);
    populateFuncToSPIRVPatterns(converter, patterns);
    populateSCFToSPIRVPatterns(converter, scfContext, patterns);
    auto legality = SPIRVConversionTarget::get(target);
    legality->addLegalOp<ModuleOp>();
    if (failed(applyFullConversion(module, *legality, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    auto converted = shader.lookupSymbol<spirv::FuncOp>(entryName);
    builder.setInsertionPointToEnd(shader.getBody());
    spirv::ExecutionModel model;
    if (stage == Stage::Vertex)
      model = spirv::ExecutionModel::Vertex;
    else if (stage == Stage::Fragment)
      model = spirv::ExecutionModel::Fragment;
    else
      model = spirv::ExecutionModel::GLCompute;
    spirv::EntryPointOp::create(builder, location, model, converted, interface);
    if (stage == Stage::Fragment)
      spirv::ExecutionModeOp::create(builder, location, converted,
                                     spirv::ExecutionMode::OriginUpperLeft,
                                     ArrayRef<int32_t>{});
    else if (stage == Stage::Compute) {
      if (wgSizeAttr)
        spirv::ExecutionModeOp::create(builder, location, converted,
                                       spirv::ExecutionMode::LocalSize,
                                       wgSizeAttr.asArrayRef());
    }
  }
};

struct GfxToLLVMPass : impl::GfxToLLVMBase<GfxToLLVMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    func::FuncOp function;
    Stage stage;
    if (failed(graphicsEntry(module, function, stage))) {
      signalPassFailure();
      return;
    }
    DenseI32ArrayAttr wgSizeAttr =
        function->getAttrOfType<DenseI32ArrayAttr>("gfx.workgroup_size");
    SmallVector<func::FuncOp> reachable;
    if (failed(collectGraphicsFunctions(module, function, reachable))) {
      signalPassFailure();
      return;
    }
    for (auto helper :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      if (!llvm::is_contained(reachable, helper))
        helper.erase();
    for (auto helper : reachable)
      if (helper != function)
        helper.setPrivate();
    OpBuilder builder(module.getContext());
    auto parameters =
        module->getAttrOfType<ParameterBlockAttr>("gfx.parameters");
    SmallVector<ParameterOp> reads;
    module.walk([&](ParameterOp op) { reads.push_back(op); });
    if (!reads.empty()) {
      auto layout = parameterLayout(parameters);
      SmallVector<Type> members;
      uint32_t offset = 0;
      for (const auto &field : layout.fields) {
        while (offset < field.offset) {
          members.push_back(builder.getF32Type());
          offset += 4;
        }
        members.push_back(field.field.getType());
        offset += field.size;
      }
      while (offset < layout.size) {
        members.push_back(builder.getF32Type());
        offset += 4;
      }
      auto structure =
          LLVM::LLVMStructType::getLiteral(module.getContext(), members, true);
      auto handleType = LLVM::LLVMTargetExtType::get(
          module.getContext(), "dx.CBuffer", ArrayRef<Type>{structure}, {});
      for (auto op : reads) {
        auto field = llvm::find_if(layout.fields, [&](const auto &field) {
          return field.field.getName() == op.getField();
        });
        builder.setInsertionPoint(op);
        auto integer = [&](uint32_t value) -> Value {
          return LLVM::ConstantOp::create(builder, op.getLoc(),
                                          builder.getI32Type(),
                                          builder.getI32IntegerAttr(value));
        };
        auto pointerType = LLVM::LLVMPointerType::get(module.getContext());
        auto null = LLVM::ZeroOp::create(builder, op.getLoc(), pointerType);
        auto handle = LLVM::CallIntrinsicOp::create(
            builder, op.getLoc(), handleType,
            builder.getStringAttr("llvm.dx.resource.handlefrombinding"),
            ValueRange{integer(0), integer(0), integer(1), integer(0), null});
        auto pointer = LLVM::CallIntrinsicOp::create(
            builder, op.getLoc(),
            LLVM::LLVMPointerType::get(module.getContext(), 2),
            builder.getStringAttr("llvm.dx.resource.getpointer"),
            ValueRange{handle.getResult(0), integer(field->offset)});
        auto value =
            LLVM::LoadOp::create(builder, op.getLoc(), field->field.getType(),
                                 pointer.getResult(0), field->alignment);
        op.getValue().replaceAllUsesWith(value);
        op.erase();
      }
    }
    module->removeAttr("gfx.parameters");
    SmallVector<Operation *> operations;
    function.walk([&](Operation *op) {
      if (isa<BuiltinOp, InputOp, OutputOp, PositionOp>(op))
        operations.push_back(op);
    });
    int32_t inputId = 0, outputId = 0;
    for (Operation *op : operations) {
      builder.setInsertionPoint(op);
      bool input = isa<BuiltinOp, InputOp>(op);
      auto integer = [&](int32_t value, unsigned width) -> Value {
        return LLVM::ConstantOp::create(
            builder, op->getLoc(), builder.getIntegerType(width),
            builder.getIntegerAttr(builder.getIntegerType(width), value));
      };
      SmallVector<Value> arguments{integer(input ? inputId++ : outputId++, 32),
                                   integer(0, 32), integer(0, 8)};
      if (input) {
        bool fragmentCoord =
            isa<BuiltinOp>(op) &&
            cast<BuiltinOp>(op).getKind() == Builtin::FragmentCoord;
        bool globalInvocationId =
            isa<BuiltinOp>(op) &&
            cast<BuiltinOp>(op).getKind() == Builtin::GlobalInvocationId;
        if (globalInvocationId) {
          Value id0 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr("llvm.dx.thread.id"),
                          ValueRange{integer(0, 32)})
                          .getResult(0);
          Value id1 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr("llvm.dx.thread.id"),
                          ValueRange{integer(1, 32)})
                          .getResult(0);
          Value id2 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr("llvm.dx.thread.id"),
                          ValueRange{integer(2, 32)})
                          .getResult(0);
          Value vec = LLVM::UndefOp::create(builder, op->getLoc(),
                                            op->getResult(0).getType());
          vec = LLVM::InsertElementOp::create(builder, op->getLoc(), vec, id0,
                                              integer(0, 32));
          vec = LLVM::InsertElementOp::create(builder, op->getLoc(), vec, id1,
                                              integer(1, 32));
          vec = LLVM::InsertElementOp::create(builder, op->getLoc(), vec, id2,
                                              integer(2, 32));
          op->getResult(0).replaceAllUsesWith(vec);
        } else {
          arguments.push_back(
              LLVM::UndefOp::create(builder, op->getLoc(), builder.getI32Type()));
          Type loadType = fragmentCoord
                              ? VectorType::get({4}, builder.getF32Type())
                              : op->getResult(0).getType();
          auto value = LLVM::CallIntrinsicOp::create(
              builder, op->getLoc(), loadType,
              builder.getStringAttr("llvm.dx.load.input"), arguments);
          Value result = value.getResult(0);
          if (fragmentCoord)
            result = vector::ShuffleOp::create(builder, op->getLoc(), result,
                                               result, ArrayRef<int64_t>{0, 1});
          op->getResult(0).replaceAllUsesWith(result);
        }
      } else {
        arguments.push_back(op->getOperand(0));
        LLVM::CallIntrinsicOp::create(
            builder, op->getLoc(),
            builder.getStringAttr("llvm.dx.store.output"), arguments);
      }
      op->erase();
    }
    function->removeAttr("gfx.stage");
    function->removeAttr("gfx.workgroup_size");
    std::string triple;
    if (stage == Stage::Vertex)
      triple = "dxil-pc-shadermodel6.6-vertex";
    else if (stage == Stage::Fragment)
      triple = "dxil-pc-shadermodel6.6-pixel";
    else
      triple = "dxil-pc-shadermodel6.6-compute";
    module->setAttr("llvm.target_triple", builder.getStringAttr(triple));
    if (stage == Stage::Compute && wgSizeAttr) {
      auto dims = wgSizeAttr.asArrayRef();
      std::string numthreads =
          llvm::formatv("{0},{1},{2}", dims[0], dims[1], dims[2]).str();
      module->setAttr("hlsl.numthreads", builder.getStringAttr(numthreads));
    }
  }
};
} // namespace

std::unique_ptr<Pass> createGfxVerifyPass() {
  return std::make_unique<GfxVerifyPass>();
}
std::unique_ptr<Pass> createGfxToSPIRVPass() {
  return std::make_unique<GfxToSPIRVPass>();
}
std::unique_ptr<Pass> createGfxToLLVMPass() {
  return std::make_unique<GfxToLLVMPass>();
}

void buildDirectXPipeline(OpPassManager &manager) {
  manager.addPass(createGfxToLLVMPass());
  // Legalize vector-valued helper boundaries without unrolling shader loops.
  manager.addPass(createInlinerPass());
  manager.addPass(createSymbolDCEPass());
  manager.addPass(createSCFToControlFlowPass());
  manager.addPass(createConvertVectorToLLVMPass());
  manager.addPass(createConvertMathToLLVMPass());
  manager.addPass(createArithToLLVMConversionPass());
  manager.addPass(createConvertControlFlowToLLVMPass());
  manager.addPass(createConvertFuncToLLVMPass());
  manager.addPass(createUBToLLVMConversionPass());
  manager.addPass(createReconcileUnrealizedCastsPass());
}

void registerGfxPipelines() {
  PassPipelineRegistration<>(
      "gfx-to-directx",
      "Lower gfx and standard operations to LLVM DirectX intrinsics",
      [](OpPassManager &manager) { buildDirectXPipeline(manager); });
}

std::unique_ptr<llvm::Module> exportDirectX(ModuleOp module,
                                            llvm::LLVMContext &context,
                                            const ShaderInterface &reflection) {
  auto translated = translateModuleToLLVMIR(module, context);
  if (!translated)
    return {};
  auto function = translated->getFunction(reflection.name);
  if (!function) {
    module.emitError("translated module is missing the reflected entry point");
    return {};
  }
  bool vertex = reflection.stage == Stage::Vertex;
  function->addFnAttr("hlsl.shader", vertex ? "vertex" : "pixel");
  if (reflection.parameters) {
    namespace rootsig = llvm::hlsl::rootsig;
    rootsig::RootConstants constants{
        parameterLayout(reflection.parameters).size / 4,
        {rootsig::RegisterType::BReg, 0}};
    SmallVector<rootsig::RootElement> elements{constants};
    auto signature =
        rootsig::MetadataBuilder(*translated, elements).BuildRootSignature();
    auto version = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 2));
    translated->getOrInsertNamedMetadata("dx.rootsignatures")
        ->addOperand(
            llvm::MDNode::get(context, {llvm::ValueAsMetadata::get(function),
                                        signature, version}));
  }
  SmallVector<llvm::Metadata *> inputs, outputs;
  auto signatures = [&](ArrayRef<InterfaceElement> descriptions,
                        SmallVectorImpl<llvm::Metadata *> &elements,
                        bool input) {
    unsigned id = 0;
    for (const auto &io : descriptions) {
      auto builtinKind = std::get_if<Builtin>(&io.binding);
      bool position = std::holds_alternative<PositionBinding>(io.binding) ||
                      (builtinKind && *builtinKind == Builtin::FragmentCoord);
      bool builtin = position || std::holds_alternative<Builtin>(io.binding);
      auto boundLocation = std::get_if<int32_t>(&io.binding);
      uint32_t location = boundLocation ? *boundLocation : 0;
      auto kind = builtin ? (position ? llvm::dxbc::PSV::SemanticKind::Position
                                      : llvm::dxbc::PSV::SemanticKind::VertexID)
                          : (!vertex && !input
                                 ? llvm::dxbc::PSV::SemanticKind::Target
                                 : llvm::dxbc::PSV::SemanticKind::Arbitrary);
      StringRef semantic = builtin
                               ? (position ? "SV_Position" : "SV_VertexID")
                               : (!vertex && !input ? "SV_Target" : "TEXCOORD");
      auto vector = dyn_cast<VectorType>(io.type);
      auto scalar = vector ? vector.getElementType() : io.type;
      auto component = scalar.isInteger(32) ? llvm::dxil::ElementType::U32
                                            : llvm::dxil::ElementType::F32;
      uint8_t columns = position ? 4 : vector ? vector.getNumElements() : 1;
      llvm::hlsl::SemanticSignatureElement element(id++, semantic, component,
                                                   kind, {location}, columns);
      element.StartRow =
          builtin ? 0 : (vertex || input ? location + 1 : location);
      element.StartCol = 0;
      auto mode = io.interpolation == Interpolation::Flat
                      ? llvm::dxbc::PSV::InterpolationMode::Constant
                  : io.interpolation == Interpolation::NoPerspective
                      ? llvm::dxbc::PSV::InterpolationMode::LinearNoperspective
                      : llvm::dxbc::PSV::InterpolationMode::Linear;
      element.InterpMode = llvm::hlsl::normalizeInterpolationMode(
          mode, component, kind,
          vertex ? llvm::Triple::Vertex : llvm::Triple::Pixel,
          input ? llvm::hlsl::IOType::In : llvm::hlsl::IOType::Out);
      elements.push_back(element.toMetadata(context));
    }
  };
  signatures(reflection.inputs, inputs, true);
  signatures(reflection.outputs, outputs, false);
  translated->getOrInsertNamedMetadata("dx.semantic.signatures")
      ->addOperand(
          llvm::MDNode::get(context, {llvm::ValueAsMetadata::get(function),
                                      llvm::MDNode::get(context, inputs),
                                      llvm::MDNode::get(context, outputs)}));
  auto integer = [&](unsigned value) -> llvm::Metadata * {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), value));
  };
  translated->getOrInsertNamedMetadata("dx.valver")
      ->addOperand(llvm::MDNode::get(context, {integer(1), integer(8)}));
  if (llvm::verifyModule(*translated, &llvm::errs())) {
    module.emitError(
        "LLVM verification rejected the translated DirectX module");
    return {};
  }
  return translated;
}
} // namespace mlir::gfx
