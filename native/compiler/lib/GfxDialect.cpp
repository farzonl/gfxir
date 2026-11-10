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
    if (!stage || stage.getValue() != Stage::Compute || !size)
      return op->emitOpError("gfx.workgroup_size requires three positive "
                             "dimensions on a compute entry point");
    TargetProfile defaultLimits;
    if (failed(defaultLimits.verifyWorkgroupSize(size.asArrayRef(), op->getLoc())))
      return failure();
  } else if (attribute.getName() == "gfx.parameters") {
    if (!isa<ModuleOp>(op) || !isa<ParameterBlockAttr>(attribute.getValue()))
      return op->emitOpError("gfx.parameters requires a typed parameter block "
                             "on the shader module");
  } else if (attribute.getName() == "gfx.required_capabilities") {
    auto array = dyn_cast<ArrayAttr>(attribute.getValue());
    if ((!isa<ModuleOp>(op) && !isa<func::FuncOp>(op)) || !array)
      return op->emitOpError(
          "gfx.required_capabilities requires an array of string attributes "
          "on a module or function");
    for (Attribute elem : array) {
      if (!isa<StringAttr>(elem))
        return op->emitOpError(
            "gfx.required_capabilities elements must be string attributes");
    }
  } else if (attribute.getName() == "gfx.shared_memory_bytes") {
    auto function = dyn_cast<func::FuncOp>(op);
    auto stage = op->getAttrOfType<StageAttr>("gfx.stage");
    auto bytes = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!function || !stage || stage.getValue() != Stage::Compute || !bytes ||
        bytes.getInt() < 0)
      return op->emitOpError(
          "gfx.shared_memory_bytes requires a non-negative integer on a "
          "compute entry point");
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

LogicalResult mlir::gfx::verifyShaderModule(ModuleOp module,
                                            const TargetProfile &profile) {
  if (failed(verify(module)))
    return failure();

  if (auto reqCapsAttr =
          module->getAttrOfType<ArrayAttr>("gfx.required_capabilities")) {
    for (auto capAttr : reqCapsAttr) {
      if (auto strAttr = dyn_cast<StringAttr>(capAttr)) {
        if (failed(profile.verifyCapability(strAttr.getValue(), module.getLoc())))
          return failure();
      }
    }
  }

  unsigned entries = 0;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (auto funcCaps =
            function->getAttrOfType<ArrayAttr>("gfx.required_capabilities")) {
      for (auto capAttr : funcCaps) {
        if (auto strAttr = dyn_cast<StringAttr>(capAttr)) {
          if (failed(profile.verifyCapability(strAttr.getValue(),
                                              function.getLoc())))
            return failure();
        }
      }
    }
    auto stage = function->getAttrOfType<StageAttr>("gfx.stage");
    if (!stage)
      continue;
    ++entries;
    if (stage.getValue() == Stage::Compute) {
      auto sizeAttr =
          function->getAttrOfType<DenseI32ArrayAttr>("gfx.workgroup_size");
      if (!sizeAttr)
        return function.emitOpError(
            "compute entry points require gfx.workgroup_size");
      if (failed(profile.verifyWorkgroupSize(sizeAttr.asArrayRef(),
                                            function.getLoc())))
        return failure();
      if (auto sharedAttr =
              function->getAttrOfType<IntegerAttr>("gfx.shared_memory_bytes")) {
        if (failed(profile.verifySharedMemory(sharedAttr.getInt(),
                                              function.getLoc())))
          return failure();
      }
    }
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

bool TargetProfile::supportsCapability(llvm::StringRef cap) const {
  return llvm::any_of(required_capabilities, [&](const std::string &item) {
    return llvm::StringRef(item).equals_insensitive(cap);
  });
}

llvm::LogicalResult
TargetProfile::verifyWorkgroupSize(llvm::ArrayRef<int32_t> dims,
                                   mlir::Location loc) const {
  if (dims.size() != 3)
    return mlir::emitError(
        loc,
        "gfx.workgroup_size requires three positive dimensions on a compute entry point");

  if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0)
    return mlir::emitError(
        loc,
        "gfx.workgroup_size requires three positive dimensions on a compute entry point");

  static const char *axes[] = {"X", "Y", "Z"};
  for (size_t i = 0; i < 3; ++i) {
    if (static_cast<uint32_t>(dims[i]) > max_workgroup_size[i]) {
      return mlir::emitError(loc)
             << "workgroup " << axes[i] << " dimension (" << dims[i]
             << ") exceeds target profile limit (" << max_workgroup_size[i]
             << ")";
    }
  }

  uint64_t total = static_cast<uint64_t>(dims[0]) *
                   static_cast<uint64_t>(dims[1]) *
                   static_cast<uint64_t>(dims[2]);
  if (total > max_workgroup_invocations) {
    return mlir::emitError(loc)
           << "total workgroup invocations (" << total
           << ") exceeds target profile limit (" << max_workgroup_invocations
           << ")";
  }

  return mlir::success();
}

llvm::LogicalResult
TargetProfile::verifySharedMemory(uint32_t bytes, mlir::Location loc) const {
  if (bytes > max_shared_memory_bytes) {
    return mlir::emitError(loc)
           << "shared memory allocation (" << bytes
           << " bytes) exceeds target limit of " << max_shared_memory_bytes
           << " bytes";
  }
  return mlir::success();
}

llvm::LogicalResult
TargetProfile::verifyCapability(llvm::StringRef cap, mlir::Location loc) const {
  if (!supportsCapability(cap)) {
    return mlir::emitError(loc)
           << "target capability profile does not support required capability '"
           << cap << "'";
  }
  return mlir::success();
}

TargetProfile TargetProfile::getDefaultVulkan() {
  TargetProfile profile;
  profile.spirv_version = 0x10000;
  profile.shader_model = 0x60;
  profile.required_capabilities = {"Shader"};
  profile.max_workgroup_size = {1024, 1024, 64};
  profile.max_workgroup_invocations = 1024;
  profile.max_shared_memory_bytes = 32768;
  return profile;
}

TargetProfile TargetProfile::getDefaultDirectX() {
  TargetProfile profile;
  profile.spirv_version = 0x10000;
  profile.shader_model = 0x60;
  profile.required_capabilities = {"Shader"};
  profile.max_workgroup_size = {1024, 1024, 64};
  profile.max_workgroup_invocations = 1024;
  profile.max_shared_memory_bytes = 32768;
  return profile;
}

std::optional<uint32_t> TargetProfile::parseSpirvVersion(llvm::StringRef str) {
  if (str.starts_with_insensitive("0x")) {
    uint32_t val = 0;
    if (!str.drop_front(2).getAsInteger(16, val))
      return val;
    return std::nullopt;
  }
  auto [majorStr, minorStr] = str.split('.');
  uint32_t major = 0, minor = 0;
  if (majorStr.getAsInteger(10, major) || minorStr.getAsInteger(10, minor))
    return std::nullopt;
  return (major << 16) | (minor << 8);
}

std::optional<uint32_t> TargetProfile::parseShaderModel(llvm::StringRef str) {
  if (str.starts_with_insensitive("0x")) {
    uint32_t val = 0;
    if (!str.drop_front(2).getAsInteger(16, val))
      return val;
    return std::nullopt;
  }
  auto [majorStr, minorStr] = str.split('.');
  uint32_t major = 0, minor = 0;
  if (majorStr.getAsInteger(10, major) || minorStr.getAsInteger(10, minor))
    return std::nullopt;
  return (major << 4) | minor;
}

std::string TargetProfile::formatSpirvVersion(uint32_t version) {
  return std::to_string((version >> 16) & 0xff) + "." +
         std::to_string((version >> 8) & 0xff);
}

std::string TargetProfile::formatShaderModel(uint32_t model) {
  return std::to_string((model >> 4) & 0xf) + "." +
         std::to_string(model & 0xf);
}

llvm::Expected<TargetProfile>
TargetProfile::fromJson(const llvm::json::Object &object) {
  TargetProfile profile;
  if (auto ver = object.getString("spirv_version")) {
    if (auto parsed = parseSpirvVersion(*ver))
      profile.spirv_version = *parsed;
    else
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid spirv_version: %s", ver->data());
  } else if (auto verInt = object.getInteger("spirv_version")) {
    profile.spirv_version = static_cast<uint32_t>(*verInt);
  }

  if (auto sm = object.getString("shader_model")) {
    if (auto parsed = parseShaderModel(*sm))
      profile.shader_model = *parsed;
    else
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid shader_model: %s", sm->data());
  } else if (auto smInt = object.getInteger("shader_model")) {
    profile.shader_model = static_cast<uint32_t>(*smInt);
  }

  if (auto caps = object.getArray("required_capabilities")) {
    profile.required_capabilities.clear();
    for (const auto &item : *caps) {
      if (auto str = item.getAsString())
        profile.required_capabilities.push_back(str->str());
    }
  }

  if (auto wg = object.getArray("max_workgroup_size")) {
    if (wg->size() != 3)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "max_workgroup_size must have 3 elements");
    for (size_t i = 0; i < 3; ++i) {
      auto val = (*wg)[i].getAsInteger();
      if (!val || *val <= 0)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "max_workgroup_size elements must be positive");
      profile.max_workgroup_size[i] = static_cast<uint32_t>(*val);
    }
  }

  if (auto inv = object.getInteger("max_workgroup_invocations")) {
    if (*inv <= 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "max_workgroup_invocations must be positive");
    profile.max_workgroup_invocations = static_cast<uint32_t>(*inv);
  }

  if (auto sh = object.getInteger("max_shared_memory_bytes")) {
    profile.max_shared_memory_bytes = static_cast<uint32_t>(*sh);
  }

  return profile;
}

llvm::json::Object TargetProfile::toJson() const {
  llvm::json::Array caps;
  for (const auto &c : required_capabilities)
    caps.push_back(c);
  llvm::json::Array wg;
  wg.push_back(static_cast<int64_t>(max_workgroup_size[0]));
  wg.push_back(static_cast<int64_t>(max_workgroup_size[1]));
  wg.push_back(static_cast<int64_t>(max_workgroup_size[2]));
  return llvm::json::Object{
      {"spirv_version", formatSpirvVersion(spirv_version)},
      {"shader_model", formatShaderModel(shader_model)},
      {"required_capabilities", std::move(caps)},
      {"max_workgroup_size", std::move(wg)},
      {"max_workgroup_invocations",
       static_cast<int64_t>(max_workgroup_invocations)},
      {"max_shared_memory_bytes", static_cast<int64_t>(max_shared_memory_bytes)},
  };
}
