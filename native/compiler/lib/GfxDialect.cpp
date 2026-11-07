#include "Gfx/Gfx.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gfx;

#include "Gfx/GfxDialect.cpp.inc"
#include "Gfx/GfxEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Gfx/GfxAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Gfx/GfxTypes.cpp.inc"
#define GET_OP_CLASSES
#include "Gfx/GfxOps.cpp.inc"

void GfxDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Gfx/GfxAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Gfx/GfxTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Gfx/GfxOps.cpp.inc"
      >();
}

void mlir::gfx::registerGfxDialects(DialectRegistry &registry) {
  registry
      .insert<GfxDialect, arith::ArithDialect, func::FuncDialect,
              vector::VectorDialect, math::MathDialect, scf::SCFDialect,
              cf::ControlFlowDialect, spirv::SPIRVDialect, LLVM::LLVMDialect>();
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  func::registerInlinerExtension(registry);
  LLVM::registerInlinerInterface(registry);
}

static StageAttr stageOf(Operation *op) {
  auto function = op->getParentOfType<func::FuncOp>();
  return function ? function->getAttrOfType<StageAttr>("gfx.stage")
                  : StageAttr{};
}

LogicalResult GfxDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attribute) {
  if (attribute.getName() == "gfx.stage") {
    auto function = dyn_cast<func::FuncOp>(op);
    if (!function || !isa<StageAttr>(attribute.getValue()))
      return op->emitOpError(
          "gfx.stage requires a typed stage attribute on func.func");
    if (!function.getFunctionType().getInputs().empty() ||
        !function.getFunctionType().getResults().empty())
      return op->emitOpError("entry points use gfx interface operations, not "
                             "host function arguments/results");
    if (function.isExternal())
      return op->emitOpError("entry points must have a body");
  } else if (attribute.getName() == "gfx.workgroup_size") {
    auto size = dyn_cast<DenseI32ArrayAttr>(attribute.getValue());
    auto stage = op->getAttrOfType<StageAttr>("gfx.stage");
    if (!stage || stage.getValue() != Stage::Compute || !size ||
        size.size() != 3 || llvm::any_of(size.asArrayRef(), [](int32_t value) {
          return value <= 0;
        }))
      return op->emitOpError("gfx.workgroup_size requires three positive "
                             "dimensions on a compute entry point");
  } else if (attribute.getName() == "gfx.parameters") {
    if (!isa<ModuleOp>(op) || !isa<ParameterBlockAttr>(attribute.getValue()))
      return op->emitOpError("gfx.parameters requires a typed parameter block "
                             "on the shader module");
  } else {
    return op->emitOpError("unknown gfx attribute: ") << attribute.getName();
  }
  return success();
}

LogicalResult
ParameterFieldAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
                           StringRef name, Type type) {
  if (!isShaderSymbolName(name))
    return error() << "parameter field requires a valid name";
  if (type.isF32())
    return success();
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 1 || !vector.getElementType().isF32() ||
      vector.getNumElements() < 2 || vector.getNumElements() > 4)
    return error() << "parameter fields currently support f32 and f32x2/3/4";
  return success();
}

static ParameterLayout layoutFields(ArrayAttr fields) {
  ParameterLayout layout{{}, 0};
  for (auto attribute : fields) {
    auto field = cast<ParameterFieldAttr>(attribute);
    auto vector = dyn_cast<VectorType>(field.getType());
    uint32_t components = vector ? vector.getNumElements() : 1;
    uint32_t alignment = components == 3 ? 16 : components * 4;
    uint32_t offset = (layout.size + alignment - 1) / alignment * alignment;
    layout.fields.push_back({field, offset, components * 4, alignment});
    layout.size = offset + components * 4;
  }
  layout.size = (layout.size + 15) / 16 * 16;
  return layout;
}

ParameterLayout mlir::gfx::parameterLayout(ParameterBlockAttr block) {
  return layoutFields(block.getFields());
}

LogicalResult
ParameterBlockAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
                           StringRef name, ArrayAttr fields) {
  if (!isShaderSymbolName(name) || fields.empty() || fields.size() > 32)
    return error() << "parameter block requires a valid name and 1-32 fields";
  llvm::StringSet<> names;
  for (auto attribute : fields) {
    auto field = dyn_cast<ParameterFieldAttr>(attribute);
    if (!field || !names.insert(field.getName()).second)
      return error() << "parameter fields must be typed and uniquely named";
    if (failed(ParameterFieldAttr::verify(error, field.getName(),
                                          field.getType())))
      return failure();
  }
  if (layoutFields(fields).size > 128)
    return error()
           << "read-only parameter blocks currently fit within 128 bytes";
  return success();
}

LogicalResult ParameterOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto block = module
                   ? module->getAttrOfType<ParameterBlockAttr>("gfx.parameters")
                   : ParameterBlockAttr{};
  if (!(*this)->getParentOfType<func::FuncOp>() || !block ||
      block.getName() != getBlock())
    return emitOpError("requires a matching declared parameter block");
  for (auto attribute : block.getFields()) {
    auto field = cast<ParameterFieldAttr>(attribute);
    if (field.getName() == getField())
      return field.getType() == getValue().getType()
                 ? success()
                 : emitOpError(
                       "parameter result type does not match its field");
  }
  return emitOpError("unknown parameter field");
}
static bool interfaceType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.getRank() != 1 || vector.getNumElements() < 1 ||
        vector.getNumElements() > 4)
      return false;
    type = vector.getElementType();
  }
  return type.isF32() || type.isInteger(32);
}

LogicalResult BuiltinOp::verify() {
  auto stage = stageOf(*this);
  if (!stage)
    return emitOpError("requires an enclosing gfx entry point");
  if (getKind() == Builtin::VertexIndex) {
    if (stage.getValue() != Stage::Vertex ||
        !getValue().getType().isInteger(32))
      return emitOpError(
          "vertex_index requires a vertex entry point and i32 result");
  } else if (getKind() == Builtin::FragmentCoord) {
    if (stage.getValue() != Stage::Fragment ||
        getValue().getType() !=
            VectorType::get({2}, Float32Type::get(getContext())))
      return emitOpError("fragment_coord requires a fragment entry point and "
                         "vector<2xf32> result");
  } else if (stage.getValue() != Stage::Compute ||
             getValue().getType() !=
                 VectorType::get({3}, IntegerType::get(getContext(), 32))) {
    return emitOpError("invocation/workgroup IDs require a compute entry point "
                       "and vector<3xi32> result");
  }
  return success();
}

static LogicalResult verifyIO(Operation *op, Type type, int32_t location,
                              Interpolation interpolation) {
  auto stage = stageOf(op);
  if (!stage || (stage.getValue() != Stage::Vertex &&
                 stage.getValue() != Stage::Fragment))
    return op->emitOpError(
        "rasterization interfaces require a vertex or fragment entry point");
  if (location < 0 || !interfaceType(type))
    return op->emitOpError("requires a nonnegative location and scalar or 1-4 "
                           "component f32/i32 type");
  auto scalar =
      isa<VectorType>(type) ? cast<VectorType>(type).getElementType() : type;
  if (scalar.isInteger(32) && interpolation != Interpolation::Flat)
    return op->emitOpError("integer interfaces require flat interpolation");
  return success();
}

LogicalResult InputOp::verify() {
  return verifyIO(*this, getValue().getType(), getLocation(),
                  getInterpolation());
}

LogicalResult OutputOp::verify() {
  return verifyIO(*this, getValue().getType(), getLocation(),
                  getInterpolation());
}

LogicalResult PositionOp::verify() {
  auto stage = stageOf(*this);
  if (!stage || stage.getValue() != Stage::Vertex ||
      getValue().getType() !=
          VectorType::get({4}, Float32Type::get(getContext())))
    return emitOpError(
        "requires a vertex entry point and vector<4xf32> clip position");
  return success();
}

LogicalResult mlir::gfx::verifyShaderModule(ModuleOp module) {
  if (failed(verify(module)))
    return failure();
  unsigned entries = 0;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto stage = function->getAttrOfType<StageAttr>("gfx.stage");
    if (!stage)
      continue;
    ++entries;
    if (stage.getValue() == Stage::Compute &&
        !function->hasAttr("gfx.workgroup_size"))
      return function.emitOpError(
          "compute entry points require gfx.workgroup_size");
    llvm::SmallDenseMap<int32_t, InputOp> inputs;
    llvm::SmallSet<int32_t, 8> outputs;
    unsigned positions = 0;
    WalkResult result = function.walk([&](Operation *op) {
      if (isa<InputOp, OutputOp, PositionOp, BuiltinOp>(op) &&
          op->getBlock() != &function.getBody().front()) {
        op->emitOpError("stage IO must be in the entry block, outside "
                        "control-flow regions");
        return WalkResult::interrupt();
      }
      if (auto input = dyn_cast<InputOp>(op)) {
        auto [existing, inserted] =
            inputs.try_emplace(input.getLocation(), input);
        if (!inserted &&
            (existing->second.getValue().getType() !=
                 input.getValue().getType() ||
             existing->second.getName() != input.getName() ||
             existing->second.getInterpolation() != input.getInterpolation())) {
          input.emitOpError("conflicting declarations for input location");
          return WalkResult::interrupt();
        }
      } else if (auto output = dyn_cast<OutputOp>(op)) {
        if (!outputs.insert(output.getLocation()).second) {
          output.emitOpError("duplicate output location");
          return WalkResult::interrupt();
        }
      } else if (isa<PositionOp>(op)) {
        ++positions;
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
    if (stage.getValue() == Stage::Vertex && positions != 1)
      return function.emitOpError(
          "vertex entry points must write exactly one clip position");
  }
  if (entries != 1)
    return module.emitError(
        "a shader compilation unit requires exactly one gfx entry point");
  return success();
}
