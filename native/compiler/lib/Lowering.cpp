#include "Gfx/Gfx.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

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
  if (stage != Stage::Vertex && stage != Stage::Fragment)
    return function.emitOpError(
        "this graphics-first lowering supports vertex and fragment stages; "
        "compute/task/mesh require additional target legalization");
  if (!function.getBody().hasOneBlock())
    return function.emitOpError(
        "this foundation supports single-block shaders; structured "
        "control-flow lowering is not implemented yet");
  return success();
}

class ReturnToSPIRV : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op);
    return success();
  }
};

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
    auto *context = module.getContext();
    auto location = function.getLoc();
    OpBuilder builder(context);
    // SPIR-V 1.0 requires a componentwise predicate for vector selects.
    function.walk([&](arith::SelectOp select) {
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
    builder.setInsertionPointToEnd(shader.getBody());
    auto converted = spirv::FuncOp::create(
        builder, location, function.getSymName(), function.getFunctionType());
    converted.getBody().takeBody(function.getBody());
    function.erase();
    SmallVector<Attribute> interface;
    SmallVector<Operation *> operations;
    converted.walk([&](Operation *op) {
      if (isa<BuiltinOp, InputOp, OutputOp, PositionOp>(op))
        operations.push_back(op);
    });
    unsigned nextSymbol = 0;
    for (Operation *op : operations) {
      bool input = isa<BuiltinOp, InputOp>(op);
      Type type =
          input ? op->getResult(0).getType() : op->getOperand(0).getType();
      auto pointerType =
          spirv::PointerType::get(type, input ? spirv::StorageClass::Input
                                              : spirv::StorageClass::Output);
      builder.setInsertionPoint(converted);
      auto variable = spirv::GlobalVariableOp::create(
          builder, op->getLoc(), pointerType,
          "interface_" + std::to_string(nextSymbol++));
      if (isa<BuiltinOp>(op))
        variable->setAttr("built_in", builder.getStringAttr("VertexIndex"));
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
        auto value = spirv::LoadOp::create(builder, op->getLoc(), address);
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
    builder.setInsertionPointToEnd(shader.getBody());
    spirv::EntryPointOp::create(builder, location,
                                stage == Stage::Vertex
                                    ? spirv::ExecutionModel::Vertex
                                    : spirv::ExecutionModel::Fragment,
                                converted, interface);
    if (stage == Stage::Fragment)
      spirv::ExecutionModeOp::create(builder, location, converted,
                                     spirv::ExecutionMode::OriginUpperLeft,
                                     ArrayRef<int32_t>{});
    SPIRVTypeConverter converter(target);
    RewritePatternSet patterns(context);
    arith::populateArithToSPIRVPatterns(converter, patterns);
    populateMathToSPIRVPatterns(converter, patterns);
    populateVectorToSPIRVPatterns(converter, patterns);
    patterns.add<ReturnToSPIRV>(context);
    auto legality = SPIRVConversionTarget::get(target);
    legality->addLegalOp<ModuleOp>();
    if (failed(applyFullConversion(module, *legality, std::move(patterns))))
      signalPassFailure();
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
    OpBuilder builder(module.getContext());
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
        arguments.push_back(
            LLVM::UndefOp::create(builder, op->getLoc(), builder.getI32Type()));
        auto value = LLVM::CallIntrinsicOp::create(
            builder, op->getLoc(), op->getResult(0).getType(),
            builder.getStringAttr("llvm.dx.load.input"), arguments);
        op->getResult(0).replaceAllUsesWith(value.getResult(0));
      } else {
        arguments.push_back(op->getOperand(0));
        LLVM::CallIntrinsicOp::create(
            builder, op->getLoc(),
            builder.getStringAttr("llvm.dx.store.output"), arguments);
      }
      op->erase();
    }
    function->removeAttr("gfx.stage");
    module->setAttr(
        "llvm.target_triple",
        builder.getStringAttr(stage == Stage::Vertex
                                  ? "dxil-pc-shadermodel6.6-vertex"
                                  : "dxil-pc-shadermodel6.6-pixel"));
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
  manager.addPass(createConvertVectorToLLVMPass());
  manager.addPass(createConvertMathToLLVMPass());
  manager.addPass(createArithToLLVMConversionPass());
  manager.addPass(createConvertFuncToLLVMPass());
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
  SmallVector<llvm::Metadata *> inputs, outputs;
  auto signatures = [&](ArrayRef<InterfaceElement> descriptions,
                        SmallVectorImpl<llvm::Metadata *> &elements,
                        bool input) {
    unsigned id = 0;
    for (const auto &io : descriptions) {
      bool position = std::holds_alternative<PositionBinding>(io.binding);
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
      uint8_t columns = vector ? vector.getNumElements() : 1;
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
