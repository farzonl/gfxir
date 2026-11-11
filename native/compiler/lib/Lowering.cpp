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
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

namespace mlir::gfx {
#define GEN_PASS_DEF_GFXTOSPIRV
#define GEN_PASS_DEF_GFXTOLLVM
#define GEN_PASS_DEF_GFXVERIFY
#include "Gfx/GfxPasses.h.inc"

namespace {
LogicalResult graphicsEntry(ModuleOp module, func::FuncOp &function,
                            Stage &stage) {
  if (failed(mlir::verify(module)))
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

static Value ensureSpirvElementPtr(ConversionPatternRewriter &rewriter,
                                   Location loc, Value ptr, Type elemType) {
  auto spvPtrType = dyn_cast<spirv::PointerType>(ptr.getType());
  if (!spvPtrType)
    return ptr;
  Type pointee = spvPtrType.getPointeeType();
  if (isa<spirv::StructType>(pointee)) {
    auto zero = spirv::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto resPtrType =
        spirv::PointerType::get(elemType, spvPtrType.getStorageClass());
    return spirv::AccessChainOp::create(rewriter, loc, resPtrType, ptr,
                                        ValueRange{zero, zero});
  }
  if (isa<spirv::ArrayType>(pointee)) {
    auto zero = spirv::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto resPtrType =
        spirv::PointerType::get(elemType, spvPtrType.getStorageClass());
    return spirv::AccessChainOp::create(rewriter, loc, resPtrType, ptr,
                                        ValueRange{zero});
  }
  return ptr;
}

struct PtrOffsetPattern : OpConversionPattern<gfx::PtrOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::PtrOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return failure();
    Value ptr = adaptor.getPtr();
    auto ptrType = dyn_cast<spirv::PointerType>(ptr.getType());
    if (ptrType && isa<spirv::StructType>(ptrType.getPointeeType())) {
      auto zero = spirv::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          op, resType, ptr, ValueRange{zero, adaptor.getOffset()});
      return success();
    }
    if (ptrType && isa<spirv::ArrayType>(ptrType.getPointeeType())) {
      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          op, resType, ptr, ValueRange{adaptor.getOffset()});
      return success();
    }
    if (auto accessChain = ptr.getDefiningOp<spirv::AccessChainOp>()) {
      auto indices = accessChain.getIndices();
      if (indices.size() == 2) {
        Value prevOffset = indices[1];
        Value addOffset = adaptor.getOffset();
        if (prevOffset.getType() != addOffset.getType()) {
          addOffset = spirv::BitcastOp::create(
              rewriter, op.getLoc(), prevOffset.getType(), addOffset);
        }
        auto combinedOffset = spirv::IAddOp::create(
            rewriter, op.getLoc(), prevOffset.getType(), prevOffset, addOffset);
        rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
            op, resType, accessChain.getBasePtr(),
            ValueRange{indices[0], combinedOffset});
        return success();
      }
      if (indices.size() == 1) {
        Value prevOffset = indices[0];
        Value addOffset = adaptor.getOffset();
        if (prevOffset.getType() != addOffset.getType()) {
          addOffset = spirv::BitcastOp::create(
              rewriter, op.getLoc(), prevOffset.getType(), addOffset);
        }
        bool isZeroBase = false;
        if (auto constOp = prevOffset.getDefiningOp<spirv::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
              intAttr && intAttr.getInt() == 0) {
            isZeroBase = true;
          }
        }
        Value combinedOffset =
            isZeroBase ? addOffset
                       : Value(spirv::IAddOp::create(rewriter, op.getLoc(),
                                                     prevOffset.getType(),
                                                     prevOffset, addOffset));
        rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
            op, resType, accessChain.getBasePtr(), ValueRange{combinedOffset});
        return success();
      }
    }
    rewriter.replaceOpWithNewOp<spirv::PtrAccessChainOp>(
        op, resType, adaptor.getPtr(), adaptor.getOffset(), ValueRange{});
    return success();
  }
};

struct LoadPattern : OpConversionPattern<gfx::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return failure();
    Value ptr = ensureSpirvElementPtr(rewriter, op.getLoc(), adaptor.getPtr(), resType);
    spirv::MemoryAccessAttr memAccess;
    IntegerAttr alignmentAttr;
    auto spvPtrType = dyn_cast<spirv::PointerType>(ptr.getType());
    bool isBDA = spvPtrType && spvPtrType.getStorageClass() ==
                                   spirv::StorageClass::PhysicalStorageBuffer;
    if (op.getAlignment().has_value()) {
      memAccess = spirv::MemoryAccessAttr::get(
          rewriter.getContext(), spirv::MemoryAccess::Aligned);
      alignmentAttr = rewriter.getI32IntegerAttr(*op.getAlignment());
    } else if (isBDA) {
      auto gfxPtrType = cast<gfx::PtrType>(op.getPtr().getType());
      uint32_t align = std::max<uint32_t>(gfxPtrType.getEffectiveAlignment(), 4u);
      memAccess = spirv::MemoryAccessAttr::get(
          rewriter.getContext(), spirv::MemoryAccess::Aligned);
      alignmentAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(align));
    }
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(
        op, resType, ptr, memAccess, alignmentAttr);
    return success();
  }
};

struct StorePattern : OpConversionPattern<gfx::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = ensureSpirvElementPtr(rewriter, op.getLoc(), adaptor.getPtr(),
                                      adaptor.getValue().getType());
    spirv::MemoryAccessAttr memAccess;
    IntegerAttr alignmentAttr;
    auto spvPtrType = dyn_cast<spirv::PointerType>(ptr.getType());
    bool isBDA = spvPtrType && spvPtrType.getStorageClass() ==
                                   spirv::StorageClass::PhysicalStorageBuffer;
    if (op.getAlignment().has_value()) {
      memAccess = spirv::MemoryAccessAttr::get(
          rewriter.getContext(), spirv::MemoryAccess::Aligned);
      alignmentAttr = rewriter.getI32IntegerAttr(*op.getAlignment());
    } else if (isBDA) {
      auto gfxPtrType = cast<gfx::PtrType>(op.getPtr().getType());
      uint32_t align = std::max<uint32_t>(gfxPtrType.getEffectiveAlignment(), 4u);
      memAccess = spirv::MemoryAccessAttr::get(
          rewriter.getContext(), spirv::MemoryAccess::Aligned);
      alignmentAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(align));
    }
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(
        op, ptr, adaptor.getValue(), memAccess, alignmentAttr);
    return success();
  }
};

struct BarrierPattern : OpConversionPattern<gfx::BarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    spirv::Scope memScope = spirv::Scope::Workgroup;
    spirv::MemorySemantics semantics =
        spirv::MemorySemantics::AcquireRelease |
        spirv::MemorySemantics::WorkgroupMemory;
    switch (op.getScope()) {
    case MemoryScope::Workgroup:
      memScope = spirv::Scope::Workgroup;
      semantics = spirv::MemorySemantics::AcquireRelease |
                  spirv::MemorySemantics::WorkgroupMemory;
      break;
    case MemoryScope::Device:
      memScope = spirv::Scope::Device;
      semantics = spirv::MemorySemantics::AcquireRelease |
                  spirv::MemorySemantics::UniformMemory;
      break;
    case MemoryScope::All:
      memScope = spirv::Scope::Device;
      semantics = spirv::MemorySemantics::AcquireRelease |
                  spirv::MemorySemantics::UniformMemory |
                  spirv::MemorySemantics::WorkgroupMemory;
      break;
    }
    rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(
        op, spirv::Scope::Workgroup, memScope, semantics);
    return success();
  }
};

struct AtomicRMWPattern : OpConversionPattern<gfx::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::AtomicRMWOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return failure();
    Value ptr =
        ensureSpirvElementPtr(rewriter, op.getLoc(), adaptor.getPtr(), resType);
    auto gfxPtrType = cast<gfx::PtrType>(op.getPtr().getType());
    spirv::Scope memScope =
        (gfxPtrType.getStorageClass() == StorageClass::Workgroup)
            ? spirv::Scope::Workgroup
            : spirv::Scope::Device;
    spirv::MemorySemantics semantics = spirv::MemorySemantics::None;
    Value val = adaptor.getValue();

    switch (op.getKind()) {
    case AtomicOp::Add:
      rewriter.replaceOpWithNewOp<spirv::AtomicIAddOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::Sub:
      rewriter.replaceOpWithNewOp<spirv::AtomicISubOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::Min:
      rewriter.replaceOpWithNewOp<spirv::AtomicSMinOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::Max:
      rewriter.replaceOpWithNewOp<spirv::AtomicSMaxOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::UMin:
      rewriter.replaceOpWithNewOp<spirv::AtomicUMinOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::UMax:
      rewriter.replaceOpWithNewOp<spirv::AtomicUMaxOp>(op, resType, ptr,
                                                       memScope, semantics, val);
      return success();
    case AtomicOp::And:
      rewriter.replaceOpWithNewOp<spirv::AtomicAndOp>(op, resType, ptr,
                                                      memScope, semantics, val);
      return success();
    case AtomicOp::Or:
      rewriter.replaceOpWithNewOp<spirv::AtomicOrOp>(op, resType, ptr, memScope,
                                                     semantics, val);
      return success();
    case AtomicOp::Xor:
      rewriter.replaceOpWithNewOp<spirv::AtomicXorOp>(op, resType, ptr,
                                                      memScope, semantics, val);
      return success();
    case AtomicOp::Xchg:
      rewriter.replaceOpWithNewOp<spirv::AtomicExchangeOp>(
          op, resType, ptr, memScope, semantics, val);
      return success();
    }
    return failure();
  }
};

struct AtomicCASPattern : OpConversionPattern<gfx::AtomicCASOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gfx::AtomicCASOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return failure();
    Value ptr =
        ensureSpirvElementPtr(rewriter, op.getLoc(), adaptor.getPtr(), resType);
    auto gfxPtrType = cast<gfx::PtrType>(op.getPtr().getType());
    spirv::Scope memScope =
        (gfxPtrType.getStorageClass() == StorageClass::Workgroup)
            ? spirv::Scope::Workgroup
            : spirv::Scope::Device;
    rewriter.replaceOpWithNewOp<spirv::AtomicCompareExchangeOp>(
        op, resType, ptr, memScope, spirv::MemorySemantics::None,
        spirv::MemorySemantics::None, adaptor.getValue(), adaptor.getCmp());
    return success();
  }
};

static bool isBoundEntryPointerRoot(Value root, func::FuncOp entryFunction) {
  if (root.getDefiningOp<BufferBindingOp>() ||
      root.getDefiningOp<SharedAllocOp>())
    return true;
  if (auto blockArg = dyn_cast<BlockArgument>(root)) {
    if (blockArg.getOwner() == &entryFunction.getBody().front()) {
      return getArgumentBufferBinding(entryFunction, blockArg.getArgNumber())
          .has_value();
    }
  }
  return false;
}

struct GfxToSPIRVPass : impl::GfxToSPIRVBase<GfxToSPIRVPass> {
  TargetProfile profile;
  GfxToSPIRVPass() = default;
  explicit GfxToSPIRVPass(const TargetProfile &profile) : profile(profile) {}

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

    bool enableBDA = profile.supportsCapability("PhysicalStorageBuffer64") ||
                     profile.supportsCapability("buffer_device_address");
    auto checkBDACapAttr = [&](ArrayAttr capsAttr) {
      if (!capsAttr)
        return;
      for (auto cap : capsAttr) {
        if (auto str = dyn_cast<StringAttr>(cap)) {
          if (str.getValue().equals_insensitive("PhysicalStorageBuffer64") ||
              str.getValue().equals_insensitive("buffer_device_address"))
            enableBDA = true;
        }
      }
    };
    checkBDACapAttr(module->getAttrOfType<ArrayAttr>("gfx.required_capabilities"));
    checkBDACapAttr(function->getAttrOfType<ArrayAttr>("gfx.required_capabilities"));

    spirv::AddressingModel addressingModel =
        enableBDA ? spirv::AddressingModel::PhysicalStorageBuffer64
                  : spirv::AddressingModel::Logical;

    bool usesStorageBuffer = false;
    bool usesVariablePointers = false;
    if (!enableBDA) {
      auto checkPtrType = [&](Type type) {
        if (auto ptrType = dyn_cast<PtrType>(type)) {
          if (ptrType.getStorageClass() == StorageClass::Global)
            usesStorageBuffer = true;
        }
      };
      for (auto func : reachable) {
        for (Type argType : func.getArgumentTypes())
          checkPtrType(argType);
        for (Type resType : func.getResultTypes())
          checkPtrType(resType);
        func.walk([&](Operation *op) {
          for (Type opType : op->getOperandTypes())
            checkPtrType(opType);
          for (Type resType : op->getResultTypes())
            checkPtrType(resType);
          if (auto offsetOp = dyn_cast<PtrOffsetOp>(op)) {
            Value root = offsetOp.getPtr();
            while (auto parentOffset = root.getDefiningOp<PtrOffsetOp>())
              root = parentOffset.getPtr();
            if (!isBoundEntryPointerRoot(root, function))
              usesVariablePointers = true;
          }
        });
      }
    }

    SmallVector<spirv::Capability, 4> capabilities = {spirv::Capability::Shader};
    SmallVector<spirv::Extension, 2> extensions;
    if (enableBDA) {
      capabilities.push_back(spirv::Capability::PhysicalStorageBufferAddresses);
      extensions.push_back(spirv::Extension::SPV_KHR_physical_storage_buffer);
    } else {
      if (usesVariablePointers) {
        capabilities.push_back(spirv::Capability::VariablePointers);
        capabilities.push_back(spirv::Capability::VariablePointersStorageBuffer);
        extensions.push_back(spirv::Extension::SPV_KHR_variable_pointers);
      }
      if (usesStorageBuffer) {
        extensions.push_back(
            spirv::Extension::SPV_KHR_storage_buffer_storage_class);
      }
    }

    auto version = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_0, capabilities, extensions, context);
    auto target = spirv::TargetEnvAttr::get(
        version, spirv::getDefaultResourceLimits(context),
        spirv::ClientAPI::Vulkan);
    auto shader = spirv::ModuleOp::create(builder, location,
                                          addressingModel,
                                          spirv::MemoryModel::GLSL450, version);
    struct EntryArgBuffer {
      unsigned argIndex;
      BufferInterfaceElement info;
    };
    SmallVector<EntryArgBuffer> entryArgBuffers;
    for (unsigned i = 0; i < function.getNumArguments(); ++i) {
      if (auto info = getArgumentBufferBinding(function, i))
        entryArgBuffers.push_back({i, *info});
    }

    function->removeAttr("gfx.stage");
    function->removeAttr("gfx.workgroup_size");
    function->removeAttr("gfx.shared_memory_bytes");
    function->removeAttr("gfx.required_capabilities");
    module->removeAttr("gfx.required_capabilities");
    for (auto helper : reachable) {
      helper->removeAttr("gfx.stage");
      helper->removeAttr("gfx.workgroup_size");
      helper->removeAttr("gfx.shared_memory_bytes");
      helper->removeAttr("gfx.required_capabilities");
      helper->moveBefore(shader.getBody(), shader.getBody()->end());
    }
    for (auto helper :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      helper.erase();
    SymbolTable symbols(shader);
    SPIRVTypeConverter converter(target);
    converter.addConversion([&](gfx::PtrType ptrType) -> std::optional<Type> {
      Type elemType = converter.convertType(ptrType.getElementType());
      if (!elemType)
        return std::nullopt;
      spirv::StorageClass sc;
      switch (ptrType.getStorageClass()) {
      case StorageClass::Global:
        sc = enableBDA ? spirv::StorageClass::PhysicalStorageBuffer
                       : spirv::StorageClass::StorageBuffer;
        break;
      case StorageClass::Workgroup:
        sc = spirv::StorageClass::Workgroup;
        break;
      case StorageClass::Private:
        sc = spirv::StorageClass::Function;
        break;
      case StorageClass::PushConstant:
        sc = spirv::StorageClass::PushConstant;
        break;
      }
      return spirv::PointerType::get(elemType, sc);
    });

    SmallVector<BufferBindingOp> bufferBindings;
    shader.walk([&](BufferBindingOp op) { bufferBindings.push_back(op); });

    auto allBlocks = getModuleParameterBlocks(module);
    spirv::GlobalVariableOp parameterVariable;
    bool readsParameters = false;
    shader.walk([&](ParameterOp) { readsParameters = true; });
    bool hasBDAPushConstants =
        enableBDA && (!entryArgBuffers.empty() || !bufferBindings.empty());

    auto createBufferBlockStructType =
        [&](Type origElemType, AccessMode access) -> spirv::StructType {
      Type elemType = converter.convertType(origElemType);
      if (!elemType)
        elemType = origElemType;
      uint32_t elemSize = 4, elemAlign = 4;
      uint32_t stride = 4;
      if (succeeded(getStd430Layout(origElemType, elemSize, elemAlign)) &&
          elemSize > 0 && elemAlign > 0) {
        stride = llvm::alignTo(elemSize, elemAlign);
      } else if (auto vecType = dyn_cast<VectorType>(elemType)) {
        stride = static_cast<uint32_t>(vecType.getNumElements()) *
                 (vecType.getElementTypeBitWidth() / 8);
      } else if (elemType.isIntOrFloat()) {
        stride = elemType.getIntOrFloatBitWidth() / 8;
      }
      auto rtArray = spirv::RuntimeArrayType::get(elemType, stride);
      SmallVector<spirv::StructType::MemberDecorationInfo, 1> memberDecos;
      if (access == AccessMode::ReadOnly) {
        memberDecos.emplace_back(0, spirv::Decoration::NonWritable,
                                 builder.getUnitAttr());
      }
      return spirv::StructType::get(
          {rtArray}, {0}, memberDecos,
          {spirv::StructType::StructDecorationInfo(spirv::Decoration::Block,
                                                   builder.getUnitAttr())});
    };

    SmallVector<uint32_t> bdaArgMemberIndices;
    SmallVector<uint32_t> bdaOpMemberIndices;
    struct ConsolidatedField {
      std::string blockName;
      std::string fieldName;
      uint32_t memberIndex;
    };
    SmallVector<ConsolidatedField> fieldLookup;

    if ((readsParameters && !allBlocks.empty()) || hasBDAPushConstants) {
      SmallVector<Type> types;
      SmallVector<uint32_t> offsets;
      uint32_t nextOffset = 0;
      if (readsParameters && !allBlocks.empty()) {
        auto toSignless = [](Type t) -> Type {
          if (auto intTy = dyn_cast<IntegerType>(t)) {
            if (!intTy.isSignless())
              return IntegerType::get(t.getContext(), intTy.getWidth());
            return t;
          }
          if (auto vecTy = dyn_cast<VectorType>(t)) {
            if (auto intElem = dyn_cast<IntegerType>(vecTy.getElementType())) {
              if (!intElem.isSignless())
                return VectorType::get(
                    vecTy.getShape(),
                    IntegerType::get(t.getContext(), intElem.getWidth()));
            }
          }
          return t;
        };
        uint32_t baseOffset = 0;
        for (auto b : allBlocks) {
          auto layout = parameterLayout(b);
          for (const auto &field : layout.fields) {
            uint32_t memberIdx = types.size();
            Type memberType = toSignless(field.field.getType());
            types.push_back(memberType);
            offsets.push_back(baseOffset + field.offset);
            fieldLookup.push_back({b.getName().str(), field.field.getName().str(), memberIdx});
          }
          baseOffset += (layout.size + 15) / 16 * 16;
        }
        nextOffset = baseOffset;
      }
      if (hasBDAPushConstants) {
        nextOffset = llvm::alignTo(nextOffset, 8u);
        for (const auto &argBuf : entryArgBuffers) {
          auto blockStruct = createBufferBlockStructType(
              argBuf.info.elementType, argBuf.info.access);
          bdaArgMemberIndices.push_back(static_cast<uint32_t>(types.size()));
          types.push_back(spirv::PointerType::get(
              blockStruct, spirv::StorageClass::PhysicalStorageBuffer));
          offsets.push_back(nextOffset);
          nextOffset += 8;
        }
        for (auto op : bufferBindings) {
          auto ptrType = cast<PtrType>(op.getResult().getType());
          auto blockStruct = createBufferBlockStructType(
              ptrType.getElementType(), op.getAccess());
          bdaOpMemberIndices.push_back(static_cast<uint32_t>(types.size()));
          types.push_back(spirv::PointerType::get(
              blockStruct, spirv::StorageClass::PhysicalStorageBuffer));
          offsets.push_back(nextOffset);
          nextOffset += 8;
        }
      }
      auto structure =
          hasBDAPushConstants
              ? spirv::StructType::get(
                    types, offsets, {},
                    {spirv::StructType::StructDecorationInfo(
                        spirv::Decoration::Block, builder.getUnitAttr())})
              : spirv::StructType::get(types, offsets);
      auto pointer =
          spirv::PointerType::get(structure, spirv::StorageClass::PushConstant);
      builder.setInsertionPoint(function);
      parameterVariable = spirv::GlobalVariableOp::create(
          builder, location, pointer, "parameters");
      symbols.insert(parameterVariable);

      if (readsParameters) {
        SmallVector<ParameterOp> reads;
        shader.walk([&](ParameterOp op) { reads.push_back(op); });
        for (auto op : reads) {
          auto found = llvm::find_if(fieldLookup, [&](const ConsolidatedField &cf) {
            return cf.blockName == op.getBlock() && cf.fieldName == op.getField();
          });
          builder.setInsertionPoint(op);
          auto address =
              spirv::AddressOfOp::create(builder, op.getLoc(), parameterVariable);
          auto index = spirv::ConstantOp::create(
              builder, op.getLoc(), builder.getI32Type(),
              builder.getI32IntegerAttr(found->memberIndex));
          auto member = spirv::AccessChainOp::create(builder, op.getLoc(),
                                                     address, ValueRange{index});
          auto value = spirv::LoadOp::create(builder, op.getLoc(), member);
          Value resVal = value.getResult();
          if (resVal.getType() != op.getValue().getType()) {
            resVal = spirv::BitcastOp::create(builder, op.getLoc(), op.getValue().getType(), resVal);
          }
          op.getValue().replaceAllUsesWith(resVal);
          op.erase();
        }
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
      if (auto builtinOp = dyn_cast<BuiltinOp>(op)) {
        StringRef builtinName = "VertexIndex";
        switch (builtinOp.getKind()) {
        case Builtin::VertexIndex:
          builtinName = "VertexIndex";
          break;
        case Builtin::FragmentCoord:
          builtinName = "FragCoord";
          break;
        case Builtin::GlobalInvocationId:
          builtinName = "GlobalInvocationId";
          break;
        case Builtin::LocalInvocationId:
          builtinName = "LocalInvocationId";
          break;
        case Builtin::WorkgroupId:
          builtinName = "WorkgroupId";
          break;
        }
        variable->setAttr("built_in", builder.getStringAttr(builtinName));
      } else if (isa<PositionOp>(op))
        variable->setAttr("built_in", builder.getStringAttr("Position"));
      else {
        int32_t index = input ? cast<InputOp>(op).getLocation()
                              : cast<OutputOp>(op).getLocation();
        Interpolation mode = input ? cast<InputOp>(op).getInterpolation()
                                   : cast<OutputOp>(op).getInterpolation();
        variable->setAttr("location", builder.getI32IntegerAttr(index));
        if (stage != Stage::Vertex || !input) {
          if (mode == Interpolation::Flat)
            variable->setAttr("flat", builder.getUnitAttr());
          else if (mode == Interpolation::NoPerspective)
            variable->setAttr("no_perspective", builder.getUnitAttr());
        }
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

    auto createStorageBufferGlobal = [&](Location loc, Type origElemType,
                                         StringRef symName, uint32_t set,
                                         uint32_t binding,
                                         AccessMode access) -> spirv::GlobalVariableOp {
      auto structType = createBufferBlockStructType(origElemType, access);
      auto varPtrType = spirv::PointerType::get(
          structType, spirv::StorageClass::StorageBuffer);
      builder.setInsertionPoint(function);
      auto var = spirv::GlobalVariableOp::create(builder, loc, varPtrType,
                                                 symName);
      var->setAttr("descriptor_set",
                   builder.getI32IntegerAttr(static_cast<int32_t>(set)));
      var->setAttr("binding",
                   builder.getI32IntegerAttr(static_cast<int32_t>(binding)));
      symbols.insert(var);
      return var;
    };

    if (!entryArgBuffers.empty()) {
      llvm::BitVector erasedArgs(function.getNumArguments());
      for (size_t idx = 0; idx < entryArgBuffers.size(); ++idx) {
        const auto &argBuf = entryArgBuffers[idx];
        erasedArgs.set(argBuf.argIndex);
        BlockArgument argVal = function.getArgument(argBuf.argIndex);
        if (enableBDA) {
          builder.setInsertionPointToStart(&function.getBody().front());
          auto address = spirv::AddressOfOp::create(builder, location,
                                                    parameterVariable);
          auto memberIdx = spirv::ConstantOp::create(
              builder, location, builder.getI32Type(),
              builder.getI32IntegerAttr(
                  static_cast<int32_t>(bdaArgMemberIndices[idx])));
          auto member = spirv::AccessChainOp::create(
              builder, location, address, ValueRange{memberIdx});
          Value bdaPtr = spirv::LoadOp::create(builder, location, member);
          argVal.replaceAllUsesWith(bdaPtr);
        } else {
          auto var = createStorageBufferGlobal(
              location, argBuf.info.elementType, argBuf.info.name,
              argBuf.info.set, argBuf.info.binding, argBuf.info.access);
          if (!argVal.use_empty()) {
            builder.setInsertionPointToStart(&function.getBody().front());
            auto addr = spirv::AddressOfOp::create(builder, location, var);
            argVal.replaceAllUsesWith(addr);
          }
        }
      }
      if (failed(function.eraseArguments(erasedArgs))) {
        signalPassFailure();
        return;
      }
    }

    for (size_t idx = 0; idx < bufferBindings.size(); ++idx) {
      auto op = bufferBindings[idx];
      if (enableBDA) {
        builder.setInsertionPoint(op);
        auto address = spirv::AddressOfOp::create(builder, op.getLoc(),
                                                  parameterVariable);
        auto memberIdx = spirv::ConstantOp::create(
            builder, op.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(
                static_cast<int32_t>(bdaOpMemberIndices[idx])));
        auto member = spirv::AccessChainOp::create(builder, op.getLoc(),
                                                   address,
                                                   ValueRange{memberIdx});
        Value bdaPtr = spirv::LoadOp::create(builder, op.getLoc(), member);
        op.getResult().replaceAllUsesWith(bdaPtr);
        op.erase();
      } else {
        auto ptrType = cast<PtrType>(op.getResult().getType());
        auto var = createStorageBufferGlobal(
            op.getLoc(), ptrType.getElementType(), op.getName(), op.getSet(),
            op.getBinding(), op.getAccess());
        builder.setInsertionPoint(op);
        auto addr = spirv::AddressOfOp::create(builder, op.getLoc(), var);
        op.getResult().replaceAllUsesWith(addr);
        op.erase();
      }
    }

    SmallVector<SharedAllocOp> sharedAllocs;
    shader.walk([&](SharedAllocOp op) { sharedAllocs.push_back(op); });
    for (size_t idx = 0; idx < sharedAllocs.size(); ++idx) {
      auto op = sharedAllocs[idx];
      auto ptrType = cast<PtrType>(op.getResult().getType());
      Type elemType = converter.convertType(ptrType.getElementType());
      if (!elemType) {
        op.emitOpError("unsupported element type for SPIR-V workgroup memory");
        signalPassFailure();
        return;
      }
      auto arrayType = spirv::ArrayType::get(elemType, op.getCount());
      auto varPtrType =
          spirv::PointerType::get(arrayType, spirv::StorageClass::Workgroup);
      std::string symName =
          (op.getName().has_value() && !op.getName()->empty())
              ? ("shared_" + op.getName()->str() + "_" + std::to_string(idx))
              : ("shared_mem_" + std::to_string(idx));
      builder.setInsertionPoint(function);
      auto var =
          spirv::GlobalVariableOp::create(builder, op.getLoc(), varPtrType, symName);
      symbols.insert(var);
      if (!op.getResult().use_empty()) {
        builder.setInsertionPoint(op);
        auto addr = spirv::AddressOfOp::create(builder, op.getLoc(), var);
        auto zero = spirv::ConstantOp::create(
            builder, op.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(0));
        auto elemPtrType =
            spirv::PointerType::get(elemType, spirv::StorageClass::Workgroup);
        auto elem0 = spirv::AccessChainOp::create(
            builder, op.getLoc(), elemPtrType, addr, ValueRange{zero});
        op.getResult().replaceAllUsesWith(elem0);
      }
      op.erase();
    }

    RewritePatternSet patterns(context);
    ScfToSPIRVContext scfContext;
    patterns.add<PtrOffsetPattern, LoadPattern, StorePattern, BarrierPattern,
                 AtomicRMWPattern, AtomicCASPattern>(converter, context);
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
    shader.walk([&](spirv::FuncOp funcOp) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        Type argType = funcOp.getArgument(i).getType();
        if (auto ptrType = dyn_cast<spirv::PointerType>(argType)) {
          if (ptrType.getStorageClass() ==
              spirv::StorageClass::PhysicalStorageBuffer) {
            funcOp.setArgAttr(
                i, spirv::DecorationAttr::name,
                spirv::DecorationAttr::get(context, spirv::Decoration::Aliased));
          }
        }
      }
    });
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
  TargetProfile profile;
  GfxToLLVMPass() = default;
  explicit GfxToLLVMPass(const TargetProfile &profile) : profile(profile) {}

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

    struct BoundBufferSlot {
      int32_t set;
      int32_t binding;
      AccessMode access;
    };
    SmallVector<BoundBufferSlot> allBufferSlots;
    auto recordSlot = [&](int32_t set, int32_t binding, AccessMode access) {
      for (const auto &existing : allBufferSlots) {
        if (existing.set == set && existing.binding == binding)
          return;
      }
      allBufferSlots.push_back({set, binding, access});
    };

    struct EntryArgBuffer {
      unsigned argIndex;
      BufferInterfaceElement info;
    };
    SmallVector<EntryArgBuffer> entryArgBuffers;
    for (unsigned i = 0; i < function.getNumArguments(); ++i) {
      if (auto info = getArgumentBufferBinding(function, i)) {
        entryArgBuffers.push_back({i, *info});
        recordSlot(info->set, info->binding, info->access);
      }
    }
    module.walk([&](BufferBindingOp bindingOp) {
      recordSlot(bindingOp.getSet(), bindingOp.getBinding(),
                 bindingOp.getAccess());
    });
    llvm::sort(allBufferSlots, [](const BoundBufferSlot &a,
                                  const BoundBufferSlot &b) {
      return std::tie(a.set, a.binding) < std::tie(b.set, b.binding);
    });
    DenseMap<std::pair<int32_t, int32_t>, uint32_t> compactRegBySlot;
    uint32_t nextUAVReg = 0;
    uint32_t nextSRVReg = 0;
    for (const auto &slot : allBufferSlots) {
      uint32_t reg =
          (slot.access == AccessMode::ReadOnly) ? nextSRVReg++ : nextUAVReg++;
      compactRegBySlot[{slot.set, slot.binding}] = reg;
    }

    auto emitRawBufferBasePtr = [&](Location loc, AccessMode access,
                                    uint32_t compactReg) -> Value {
      unsigned int isWriteable = (access == AccessMode::ReadOnly) ? 0u : 1u;
      unsigned int isROV = 0u;
      auto handleType = LLVM::LLVMTargetExtType::get(
          module.getContext(), "dx.RawBuffer",
          ArrayRef<Type>{builder.getI8Type()},
          ArrayRef<unsigned int>{isWriteable, isROV});
      auto integer = [&](uint32_t value) -> Value {
        return LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                        builder.getI32IntegerAttr(value));
      };
      auto pointerType = LLVM::LLVMPointerType::get(module.getContext());
      auto nullVal = LLVM::ZeroOp::create(builder, loc, pointerType);
      auto handle = LLVM::CallIntrinsicOp::create(
          builder, loc, handleType,
          builder.getStringAttr("llvm.dx.resource.handlefrombinding"),
          ValueRange{integer(0), integer(compactReg), integer(1), integer(0),
                     nullVal});
      auto resPtrType = LLVM::LLVMPointerType::get(module.getContext(), 1);
      auto pointer = LLVM::CallIntrinsicOp::create(
          builder, loc, resPtrType,
          builder.getStringAttr("llvm.dx.resource.getpointer"),
          ValueRange{handle.getResult(0), integer(0)});
      return pointer.getResult(0);
    };

    if (!entryArgBuffers.empty()) {
      llvm::BitVector erasedArgs(function.getNumArguments());
      for (const auto &argBuf : entryArgBuffers) {
        erasedArgs.set(argBuf.argIndex);
        BlockArgument argVal = function.getArgument(argBuf.argIndex);
        if (!argVal.use_empty()) {
          builder.setInsertionPointToStart(&function.getBody().front());
          uint32_t compactReg =
              compactRegBySlot[{argBuf.info.set, argBuf.info.binding}];
          Value basePtr = emitRawBufferBasePtr(function.getLoc(),
                                               argBuf.info.access, compactReg);
          Value castPtr = UnrealizedConversionCastOp::create(
                              builder, function.getLoc(),
                              TypeRange{argVal.getType()}, ValueRange{basePtr})
                              .getResult(0);
          argVal.replaceAllUsesWith(castPtr);
        }
      }
      if (failed(function.eraseArguments(erasedArgs))) {
        signalPassFailure();
        return;
      }
    }

    std::function<Type(Type)> convertGfxPtr = [&](Type t) -> Type {
      if (auto ptr = dyn_cast<gfx::PtrType>(t)) {
        unsigned addrSpace = 0;
        switch (ptr.getStorageClass()) {
        case StorageClass::Private: addrSpace = 0; break;
        case StorageClass::Global: addrSpace = 1; break;
        case StorageClass::PushConstant: addrSpace = 2; break;
        case StorageClass::Workgroup: addrSpace = 3; break;
        }
        return LLVM::LLVMPointerType::get(module.getContext(), addrSpace);
      }
      if (auto intType = dyn_cast<IntegerType>(t)) {
        if (!intType.isSignless())
          return builder.getIntegerType(intType.getWidth());
      }
      if (auto vecType = dyn_cast<VectorType>(t)) {
        Type convertedElem = convertGfxPtr(vecType.getElementType());
        if (convertedElem != vecType.getElementType())
          return VectorType::get(vecType.getShape(), convertedElem);
      }
      return t;
    };
    for (auto fn : reachable) {
      SmallVector<Type> newArgTypes;
      for (auto arg : fn.getArguments()) {
        Type newType = convertGfxPtr(arg.getType());
        if (newType != arg.getType())
          arg.setType(newType);
        newArgTypes.push_back(arg.getType());
      }
      SmallVector<Type> newResTypes;
      for (Type resType : fn.getResultTypes())
        newResTypes.push_back(convertGfxPtr(resType));
      fn.setType(builder.getFunctionType(newArgTypes, newResTypes));
    }
    auto allBlocks = getModuleParameterBlocks(module);
    SmallVector<ParameterOp> reads;
    module.walk([&](ParameterOp op) { reads.push_back(op); });
    if (!reads.empty() && !allBlocks.empty()) {
      struct BlockHandleInfo {
        ParameterBlockAttr block;
        ParameterLayout layout;
        Type handleType;
        uint32_t blockIdx;
      };
      SmallVector<BlockHandleInfo> blockInfos;
      for (size_t i = 0; i < allBlocks.size(); ++i) {
        auto b = allBlocks[i];
        auto l = parameterLayout(b);
        SmallVector<Type> members;
        uint32_t offset = 0;
        for (const auto &field : l.fields) {
          while (offset < field.offset) {
            members.push_back(builder.getF32Type());
            offset += 4;
          }
          Type memberType = convertGfxPtr(field.field.getType());
          members.push_back(memberType);
          offset += field.size;
        }
        while (offset < l.size) {
          members.push_back(builder.getF32Type());
          offset += 4;
        }
        auto structure =
            LLVM::LLVMStructType::getLiteral(module.getContext(), members, true);
        auto handleType = LLVM::LLVMTargetExtType::get(
            module.getContext(), "dx.CBuffer", ArrayRef<Type>{structure}, {});
        blockInfos.push_back({b, l, handleType, static_cast<uint32_t>(i)});
      }

      for (auto op : reads) {
        auto infoIt = llvm::find_if(blockInfos, [&](const BlockHandleInfo &info) {
          return info.block.getName() == op.getBlock();
        });
        if (infoIt == blockInfos.end())
          continue;
        const auto &info = *infoIt;
        auto field = llvm::find_if(info.layout.fields, [&](const auto &f) {
          return f.field.getName() == op.getField();
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
            builder, op.getLoc(), info.handleType,
            builder.getStringAttr("llvm.dx.resource.handlefrombinding"),
            ValueRange{integer(0), integer(info.blockIdx), integer(1), integer(0), null});
        auto pointer = LLVM::CallIntrinsicOp::create(
            builder, op.getLoc(),
            LLVM::LLVMPointerType::get(module.getContext(), 2),
            builder.getStringAttr("llvm.dx.resource.getpointer"),
            ValueRange{handle.getResult(0), integer(field->offset)});
        Type loadType = convertGfxPtr(field->field.getType());
        auto value =
            LLVM::LoadOp::create(builder, op.getLoc(), loadType,
                                 pointer.getResult(0), field->alignment);
        Value resVal = value.getResult();
        if (resVal.getType() != op.getValue().getType()) {
          resVal = UnrealizedConversionCastOp::create(
              builder, op.getLoc(), TypeRange{op.getValue().getType()},
              ValueRange{resVal}).getResult(0);
        }
        op.getValue().replaceAllUsesWith(resVal);
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
        StringRef computeIntrinsic;
        if (auto builtinOp = dyn_cast<BuiltinOp>(op)) {
          switch (builtinOp.getKind()) {
          case Builtin::GlobalInvocationId:
            computeIntrinsic = "llvm.dx.thread.id";
            break;
          case Builtin::LocalInvocationId:
            computeIntrinsic = "llvm.dx.thread.id.in.group";
            break;
          case Builtin::WorkgroupId:
            computeIntrinsic = "llvm.dx.group.id";
            break;
          default:
            break;
          }
        }
        if (!computeIntrinsic.empty()) {
          Value id0 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr(computeIntrinsic),
                          ValueRange{integer(0, 32)})
                          .getResult(0);
          Value id1 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr(computeIntrinsic),
                          ValueRange{integer(1, 32)})
                          .getResult(0);
          Value id2 = LLVM::CallIntrinsicOp::create(
                          builder, op->getLoc(), builder.getI32Type(),
                          builder.getStringAttr(computeIntrinsic),
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
    SmallVector<SharedAllocOp> sharedAllocs;
    module.walk([&](SharedAllocOp op) { sharedAllocs.push_back(op); });
    for (size_t idx = 0; idx < sharedAllocs.size(); ++idx) {
      auto allocOp = sharedAllocs[idx];
      auto ptrType = cast<PtrType>(allocOp.getResult().getType());
      Type elemType = convertGfxPtr(ptrType.getElementType());
      auto arrayType =
          LLVM::LLVMArrayType::get(elemType, static_cast<unsigned>(allocOp.getCount()));
      uint32_t align =
          std::max<uint32_t>(getNaturalAlignment(ptrType.getElementType()), 4u);
      std::string symName =
          (allocOp.getName().has_value() && !allocOp.getName()->empty())
              ? ("shared_" + allocOp.getName()->str() + "_" +
                 std::to_string(idx))
              : ("shared_mem_" + std::to_string(idx));
      builder.setInsertionPointToStart(module.getBody());
      auto globalOp = LLVM::GlobalOp::create(
          builder, allocOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, symName, /*value=*/Attribute{},
          /*alignment=*/align, /*addrSpace=*/3);
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *initBlock =
            builder.createBlock(&globalOp.getInitializerRegion());
        builder.setInsertionPointToStart(initBlock);
        Value undefVal =
            LLVM::UndefOp::create(builder, allocOp.getLoc(), arrayType);
        LLVM::ReturnOp::create(builder, allocOp.getLoc(), ValueRange{undefVal});
      }
      if (!allocOp.getResult().use_empty()) {
        builder.setInsertionPoint(allocOp);
        auto addr =
            LLVM::AddressOfOp::create(builder, allocOp.getLoc(), globalOp);
        Value castPtr =
            UnrealizedConversionCastOp::create(
                builder, allocOp.getLoc(),
                TypeRange{allocOp.getResult().getType()},
                ValueRange{addr.getRes()})
                .getResult(0);
        allocOp.getResult().replaceAllUsesWith(castPtr);
      }
      allocOp.erase();
    }

    SmallVector<Operation *> bufferOps;
    module.walk([&](Operation *op) {
      if (isa<BufferBindingOp, PtrOffsetOp, LoadOp, StoreOp, BarrierOp,
              AtomicRMWOp, AtomicCASOp>(op))
        bufferOps.push_back(op);
    });
    for (Operation *op : bufferOps) {
      builder.setInsertionPoint(op);
      if (auto binding = dyn_cast<BufferBindingOp>(op)) {
        uint32_t compactReg =
            compactRegBySlot[{binding.getSet(), binding.getBinding()}];
        Value pointer =
            emitRawBufferBasePtr(op->getLoc(), binding.getAccess(), compactReg);
        op->getResult(0).replaceAllUsesWith(pointer);
        op->erase();
      } else if (auto barrier = dyn_cast<BarrierOp>(op)) {
        StringRef intrinsicName = "llvm.dx.group.memory.barrier.with.group.sync";
        switch (barrier.getScope()) {
        case MemoryScope::Workgroup:
          intrinsicName = "llvm.dx.group.memory.barrier.with.group.sync";
          break;
        case MemoryScope::Device:
          intrinsicName = "llvm.dx.device.memory.barrier.with.group.sync";
          break;
        case MemoryScope::All:
          intrinsicName = "llvm.dx.all.memory.barrier.with.group.sync";
          break;
        }
        LLVM::CallIntrinsicOp::create(
            builder, op->getLoc(), builder.getStringAttr(intrinsicName),
            ValueRange{});
        op->erase();
      } else if (auto rmw = dyn_cast<AtomicRMWOp>(op)) {
        Value ptr = op->getOperand(0);
        Type expectedPtrType = convertGfxPtr(ptr.getType());
        if (ptr.getType() != expectedPtrType) {
          ptr = UnrealizedConversionCastOp::create(
                    builder, op->getLoc(), TypeRange{expectedPtrType},
                    ValueRange{ptr})
                    .getResult(0);
        }
        Value val = op->getOperand(1);
        Type expectedElem = convertGfxPtr(val.getType());
        if (val.getType() != expectedElem) {
          val = UnrealizedConversionCastOp::create(
                    builder, op->getLoc(), TypeRange{expectedElem},
                    ValueRange{val})
                    .getResult(0);
        }
        LLVM::AtomicBinOp binOp = LLVM::AtomicBinOp::add;
        switch (rmw.getKind()) {
        case AtomicOp::Add:
          binOp = LLVM::AtomicBinOp::add;
          break;
        case AtomicOp::Sub: {
          // DXIL RawBuffer atomics lower InterlockedAdd with a negated operand
          // instead of a separate subtract opcode.
          Value zero = LLVM::ConstantOp::create(
              builder, op->getLoc(), expectedElem,
              builder.getIntegerAttr(expectedElem, 0));
          val = LLVM::SubOp::create(builder, op->getLoc(), zero, val);
          binOp = LLVM::AtomicBinOp::add;
          break;
        }
        case AtomicOp::Min:
          binOp = LLVM::AtomicBinOp::min;
          break;
        case AtomicOp::Max:
          binOp = LLVM::AtomicBinOp::max;
          break;
        case AtomicOp::UMin:
          binOp = LLVM::AtomicBinOp::umin;
          break;
        case AtomicOp::UMax:
          binOp = LLVM::AtomicBinOp::umax;
          break;
        case AtomicOp::And:
          binOp = LLVM::AtomicBinOp::_and;
          break;
        case AtomicOp::Or:
          binOp = LLVM::AtomicBinOp::_or;
          break;
        case AtomicOp::Xor:
          binOp = LLVM::AtomicBinOp::_xor;
          break;
        case AtomicOp::Xchg:
          binOp = LLVM::AtomicBinOp::xchg;
          break;
        }
        auto llvmRmw = LLVM::AtomicRMWOp::create(
            builder, op->getLoc(), binOp, ptr, val,
            LLVM::AtomicOrdering::seq_cst, StringRef(), /*alignment=*/4u);
        Value resVal = llvmRmw.getRes();
        if (resVal.getType() != rmw.getResult().getType()) {
          resVal = UnrealizedConversionCastOp::create(
                       builder, op->getLoc(),
                       TypeRange{rmw.getResult().getType()}, ValueRange{resVal})
                       .getResult(0);
        }
        rmw.getResult().replaceAllUsesWith(resVal);
        op->erase();
      } else if (auto cas = dyn_cast<AtomicCASOp>(op)) {
        Value ptr = op->getOperand(0);
        Type expectedPtrType = convertGfxPtr(ptr.getType());
        if (ptr.getType() != expectedPtrType) {
          ptr = UnrealizedConversionCastOp::create(
                    builder, op->getLoc(), TypeRange{expectedPtrType},
                    ValueRange{ptr})
                    .getResult(0);
        }
        Value cmp = op->getOperand(1);
        Value val = op->getOperand(2);
        Type expectedElem = convertGfxPtr(cmp.getType());
        if (cmp.getType() != expectedElem) {
          cmp = UnrealizedConversionCastOp::create(
                    builder, op->getLoc(), TypeRange{expectedElem},
                    ValueRange{cmp})
                    .getResult(0);
        }
        if (val.getType() != expectedElem) {
          val = UnrealizedConversionCastOp::create(
                    builder, op->getLoc(), TypeRange{expectedElem},
                    ValueRange{val})
                    .getResult(0);
        }
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            builder, op->getLoc(), ptr, cmp, val,
            LLVM::AtomicOrdering::seq_cst, LLVM::AtomicOrdering::seq_cst,
            StringAttr{}, builder.getI64IntegerAttr(4), /*weak=*/false,
            /*volatile_=*/false, ArrayAttr{}, ArrayAttr{}, ArrayAttr{},
            ArrayAttr{});
        Value resVal = LLVM::ExtractValueOp::create(
            builder, op->getLoc(), cmpxchg.getRes(), ArrayRef<int64_t>{0});
        if (resVal.getType() != cas.getResult().getType()) {
          resVal = UnrealizedConversionCastOp::create(
                       builder, op->getLoc(),
                       TypeRange{cas.getResult().getType()}, ValueRange{resVal})
                       .getResult(0);
        }
        cas.getResult().replaceAllUsesWith(resVal);
        op->erase();
      } else if (isa<PtrOffsetOp>(op)) {
        Value ptr = op->getOperand(0);
        Value offsetVal = op->getOperand(1);
        if (isa<IndexType>(offsetVal.getType())) {
          offsetVal = arith::IndexCastOp::create(
              builder, op->getLoc(), builder.getI32Type(), offsetVal);
        } else if (auto intType = dyn_cast<IntegerType>(offsetVal.getType())) {
          bool isUnsigned = intType.isUnsigned();
          unsigned width = intType.getWidth();
          if (!intType.isSignless()) {
            Type signlessType = builder.getIntegerType(width);
            offsetVal = UnrealizedConversionCastOp::create(
                builder, op->getLoc(), TypeRange{signlessType},
                ValueRange{offsetVal}).getResult(0);
          }
          if (width > 32) {
            offsetVal = LLVM::TruncOp::create(
                builder, op->getLoc(), builder.getI32Type(), offsetVal);
          } else if (width < 32) {
            if (isUnsigned) {
              offsetVal = LLVM::ZExtOp::create(
                  builder, op->getLoc(), builder.getI32Type(), offsetVal);
            } else {
              offsetVal = LLVM::SExtOp::create(
                  builder, op->getLoc(), builder.getI32Type(), offsetVal);
            }
          }
        }
        auto gfxPtrType = cast<PtrType>(op->getResult(0).getType());
        Type elemType = convertGfxPtr(gfxPtrType.getElementType());
        Type resPtrType = convertGfxPtr(gfxPtrType);
        if (ptr.getType() != resPtrType) {
          ptr = UnrealizedConversionCastOp::create(
              builder, op->getLoc(), TypeRange{resPtrType},
              ValueRange{ptr}).getResult(0);
        }
        auto gep = LLVM::GEPOp::create(
            builder, op->getLoc(), resPtrType, elemType,
            ptr, ArrayRef<LLVM::GEPArg>{offsetVal});
        op->getResult(0).replaceAllUsesWith(gep);
        op->erase();
      } else if (auto load = dyn_cast<LoadOp>(op)) {
        Value ptr = op->getOperand(0);
        Type expectedPtrType = convertGfxPtr(ptr.getType());
        if (ptr.getType() != expectedPtrType) {
          ptr = UnrealizedConversionCastOp::create(
              builder, op->getLoc(), TypeRange{expectedPtrType},
              ValueRange{ptr}).getResult(0);
        }
        Type elemType = convertGfxPtr(load.getResult().getType());
        uint64_t align = load.getAlignment().value_or(0);
        auto llvmLoad = LLVM::LoadOp::create(
            builder, op->getLoc(), elemType, ptr, align);
        Value resVal = llvmLoad.getResult();
        if (resVal.getType() != load.getResult().getType()) {
          resVal = UnrealizedConversionCastOp::create(
              builder, op->getLoc(), TypeRange{load.getResult().getType()},
              ValueRange{resVal}).getResult(0);
        }
        load.getResult().replaceAllUsesWith(resVal);
        op->erase();
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        Value val = op->getOperand(0);
        Value ptr = op->getOperand(1);
        uint64_t align = store.getAlignment().value_or(0);
        Type expectedElem = convertGfxPtr(val.getType());
        if (val.getType() != expectedElem) {
          val = UnrealizedConversionCastOp::create(
              builder, op->getLoc(), TypeRange{expectedElem},
              ValueRange{val}).getResult(0);
        }
        Type expectedPtrType = convertGfxPtr(ptr.getType());
        if (ptr.getType() != expectedPtrType) {
          ptr = UnrealizedConversionCastOp::create(
              builder, op->getLoc(), TypeRange{expectedPtrType},
              ValueRange{ptr}).getResult(0);
        }
        LLVM::StoreOp::create(
            builder, op->getLoc(), val, ptr, align);
        op->erase();
      }
    }
    function->removeAttr("gfx.stage");
    function->removeAttr("gfx.workgroup_size");
    function->removeAttr("gfx.shared_memory_bytes");
    function->removeAttr("gfx.required_capabilities");
    module->removeAttr("gfx.required_capabilities");
    std::string triple;
    if (stage == Stage::Vertex)
      triple = "dxil-pc-shadermodel6.6-vertex";
    else if (stage == Stage::Fragment)
      triple = "dxil-pc-shadermodel6.6-pixel";
    else
      triple = "dxil-pc-shadermodel6.6-compute";
    module->setAttr("llvm.target_triple", builder.getStringAttr(triple));
    if (stage == Stage::Compute) {
      std::string numthreads = "1,1,1";
      if (wgSizeAttr) {
        auto dims = wgSizeAttr.asArrayRef();
        numthreads =
            llvm::formatv("{0},{1},{2}", dims[0], dims[1], dims[2]).str();
      }
      module->setAttr("hlsl.numthreads", builder.getStringAttr(numthreads));
      SmallVector<Attribute, 2> passthroughAttrs{
          builder.getArrayAttr({builder.getStringAttr("hlsl.shader"),
                                builder.getStringAttr("compute")}),
          builder.getArrayAttr({builder.getStringAttr("hlsl.numthreads"),
                                builder.getStringAttr(numthreads)}),
      };
      function->setAttr("llvm.passthrough",
                        builder.getArrayAttr(passthroughAttrs));
    }
  }
};
} // namespace

std::unique_ptr<Pass> createGfxVerifyPass() {
  return std::make_unique<GfxVerifyPass>();
}
std::unique_ptr<Pass> createGfxToSPIRVPass(const TargetProfile &profile) {
  return std::make_unique<GfxToSPIRVPass>(profile);
}
std::unique_ptr<Pass> createGfxToLLVMPass(const TargetProfile &profile) {
  return std::make_unique<GfxToLLVMPass>(profile);
}

void buildDirectXPipeline(OpPassManager &manager, const TargetProfile &profile) {
  manager.addPass(createGfxToLLVMPass(profile));
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
  PassPipelineRegistration<>(
      "gfx-to-vulkan",
      "Lower gfx and standard operations to SPIR-V Vulkan",
      [](OpPassManager &manager) { manager.addPass(createGfxToSPIRVPass()); });
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
  const char *shaderKind = "pixel";
  if (reflection.stage == Stage::Vertex) {
    shaderKind = "vertex";
  } else if (reflection.stage == Stage::Compute) {
    shaderKind = "compute";
    if (auto wgAttr = module->getAttrOfType<DenseI32ArrayAttr>("gfx.workgroup_size")) {
      auto dims = wgAttr.asArrayRef();
      std::string numthreads = llvm::formatv("{0},{1},{2}", dims[0], dims[1], dims[2]).str();
      function->addFnAttr("hlsl.numthreads", numthreads);
    } else if (auto numthreadsAttr = module->getAttrOfType<StringAttr>("hlsl.numthreads")) {
      function->addFnAttr("hlsl.numthreads", numthreadsAttr.getValue());
    } else {
      function->addFnAttr("hlsl.numthreads", "1,1,1");
    }
  }
  function->addFnAttr("hlsl.shader", shaderKind);
  bool vertex = reflection.stage == Stage::Vertex;
  SmallVector<llvm::hlsl::rootsig::RootElement> elements;
  if (!reflection.parameterBlocks.empty()) {
    for (const auto &pBlock : reflection.parameterBlocks) {
      namespace rootsig = llvm::hlsl::rootsig;
      rootsig::RootConstants constants{
          parameterLayout(pBlock.block).size / 4,
          {rootsig::RegisterType::BReg, pBlock.registerIndex}};
      elements.push_back(constants);
    }
  } else if (reflection.parameters) {
    namespace rootsig = llvm::hlsl::rootsig;
    rootsig::RootConstants constants{
        (parameterLayout(reflection.parameters).size + 3) / 4,
        {rootsig::RegisterType::BReg, 0}};
    elements.push_back(constants);
  }
  if (!reflection.buffers.empty()) {
    namespace rootsig = llvm::hlsl::rootsig;
    SmallVector<BufferInterfaceElement> uniqueBuffers;
    for (const auto &buf : reflection.buffers) {
      bool exists = llvm::any_of(uniqueBuffers, [&](const auto &existing) {
        return existing.set == buf.set && existing.binding == buf.binding;
      });
      if (!exists)
        uniqueBuffers.push_back(buf);
    }
    uint32_t uavCount = 0;
    uint32_t srvCount = 0;
    for (const auto &buf : uniqueBuffers) {
      if (buf.access == AccessMode::ReadOnly)
        ++srvCount;
      else
        ++uavCount;
    }
    uint32_t numClauses = 0;
    if (uavCount > 0) {
      rootsig::DescriptorTableClause uavClause;
      uavClause.Type = llvm::dxil::ResourceClass::UAV;
      uavClause.Reg = rootsig::Register{rootsig::RegisterType::UReg, 0};
      uavClause.NumDescriptors = uavCount;
      uavClause.Space = 0;
      uavClause.setDefaultFlags(llvm::dxbc::RootSignatureVersion::V1_1);
      elements.push_back(uavClause);
      ++numClauses;
    }
    if (srvCount > 0) {
      rootsig::DescriptorTableClause srvClause;
      srvClause.Type = llvm::dxil::ResourceClass::SRV;
      srvClause.Reg = rootsig::Register{rootsig::RegisterType::TReg, 0};
      srvClause.NumDescriptors = srvCount;
      srvClause.Space = 0;
      srvClause.setDefaultFlags(llvm::dxbc::RootSignatureVersion::V1_1);
      elements.push_back(srvClause);
      ++numClauses;
    }
    if (numClauses > 0) {
      rootsig::DescriptorTable table;
      table.Visibility = llvm::dxbc::ShaderVisibility::All;
      table.NumClauses = numClauses;
      elements.push_back(table);
    }
  }
  if (!elements.empty()) {
    namespace rootsig = llvm::hlsl::rootsig;
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
  if (reflection.stage != Stage::Compute) {
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
        std::string semanticStr;
        uint32_t semanticIndex = location;
        if (builtin) {
          semanticStr = position ? "SV_Position" : "SV_VertexID";
        } else if (!vertex && !input) {
          semanticStr = "SV_Target";
        } else if (vertex && input) {
          StringRef name = io.name;
          if (name.equals_insensitive("position") || name.equals_insensitive("pos")) {
            semanticStr = "POSITION";
            semanticIndex = 0;
          } else if (name.equals_insensitive("normal") || name.equals_insensitive("norm")) {
            semanticStr = "NORMAL";
            semanticIndex = 0;
          } else if (name.equals_insensitive("uv") || name.starts_with_insensitive("texcoord")) {
            semanticStr = "TEXCOORD";
            semanticIndex = 0;
          } else {
            semanticStr = name.upper();
            semanticIndex = 0;
          }
        } else {
          semanticStr = "TEXCOORD";
        }
        StringRef semantic = semanticStr;
        auto vector = dyn_cast<VectorType>(io.type);
        auto scalar = vector ? vector.getElementType() : io.type;
        auto component = scalar.isInteger(32) ? llvm::dxil::ElementType::U32
                                              : llvm::dxil::ElementType::F32;
        uint8_t columns = position ? 4 : vector ? vector.getNumElements() : 1;
        llvm::hlsl::SemanticSignatureElement element(id++, semantic, component,
                                                     kind, {semanticIndex}, columns);
        if (builtin) {
          element.StartRow = 0;
        } else if (vertex && input) {
          element.StartRow = location;
        } else if (!vertex && !input) {
          element.StartRow = location;
        } else {
          element.StartRow = location + 1;
        }
        element.StartCol = 0;
        auto mode = io.interpolation == Interpolation::Flat
                        ? llvm::dxbc::PSV::InterpolationMode::Constant
                    : io.interpolation == Interpolation::NoPerspective
                        ? llvm::dxbc::PSV::InterpolationMode::LinearNoperspective
                        : llvm::dxbc::PSV::InterpolationMode::Linear;
        llvm::Triple::EnvironmentType stageEnv =
            vertex ? llvm::Triple::Vertex
                   : (reflection.stage == Stage::Compute ? llvm::Triple::Compute
                                                         : llvm::Triple::Pixel);
        element.InterpMode = llvm::hlsl::normalizeInterpolationMode(
            mode, component, kind, stageEnv,
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
  }
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
