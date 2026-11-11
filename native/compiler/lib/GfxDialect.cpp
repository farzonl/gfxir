#include "Gfx/Gfx.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

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

uint32_t mlir::gfx::getTypeSize(Type type) {
  if (!type)
    return 0;
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16() || type.isInteger(16))
    return 2;
  if (type.isF64() || type.isInteger(64))
    return 8;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.getRank() == 1)
      return vector.getNumElements() * getTypeSize(vector.getElementType());
  }
  return 0;
}

//=============================================================================
// std430 Natural Alignment with Cycle Safety
//=============================================================================
static uint32_t getNaturalAlignmentImpl(Type type, llvm::SmallPtrSetImpl<Type> &visiting) {
  if (!type)
    return 0;
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16() || type.isInteger(16))
    return 2;
  if (type.isF64() || type.isInteger(64))
    return 8;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.getRank() != 1)
      return 0;
    uint32_t elemAlign = getNaturalAlignmentImpl(vector.getElementType(), visiting);
    if (elemAlign == 0)
      return 0;
    uint32_t numElements = vector.getNumElements();
    // std430 alignment rules: vec2 = 2 * scalar, vec3 and vec4 = 4 * scalar
    uint32_t factor = (numElements == 3 || numElements == 4) ? 4 : numElements;
    return factor * elemAlign;
  }

  // Cycle detection: recursive by-value struct has no finite alignment
  if (!visiting.insert(type).second)
    return 0;

  uint32_t maxAlign = 0;
  if (auto spirvStruct = dyn_cast<spirv::StructType>(type)) {
    maxAlign = 1;
    for (Type member : spirvStruct.getElementTypes()) {
      uint32_t memberAlign = getNaturalAlignmentImpl(member, visiting);
      if (memberAlign == 0) {
        maxAlign = 0;
        break;
      }
      maxAlign = std::max(maxAlign, memberAlign);
    }
  } else if (auto llvmStruct = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (!llvmStruct.isOpaque()) {
      maxAlign = 1;
      for (Type member : llvmStruct.getBody()) {
        uint32_t memberAlign = getNaturalAlignmentImpl(member, visiting);
        if (memberAlign == 0) {
          maxAlign = 0;
          break;
        }
        maxAlign = std::max(maxAlign, memberAlign);
      }
    }
  } else if (auto tuple = dyn_cast<TupleType>(type)) {
    maxAlign = 1;
    for (Type member : tuple.getTypes()) {
      uint32_t memberAlign = getNaturalAlignmentImpl(member, visiting);
      if (memberAlign == 0) {
        maxAlign = 0;
        break;
      }
      maxAlign = std::max(maxAlign, memberAlign);
    }
  }

  visiting.erase(type);
  return maxAlign;
}

uint32_t mlir::gfx::getNaturalAlignment(Type type) {
  llvm::SmallPtrSet<Type, 8> visiting;
  return getNaturalAlignmentImpl(type, visiting);
}

//=============================================================================
// Recursive Pointee Type Validation with Cycle Detection & Memoization
//=============================================================================
static bool isValidPointeeTypeImpl(Type type,
                                   llvm::SmallPtrSetImpl<Type> &visiting,
                                   llvm::SmallPtrSetImpl<Type> &visited) {
  if (!type)
    return false;
  // Disallow nested pointers (gfx, llvm, and spirv pointers)
  if (isa<PtrType, LLVM::LLVMPointerType, spirv::PointerType>(type))
    return false;
  // Disallow opaque resource handles
  if (isa<HandleType>(type))
    return false;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.getRank() != 1)
      return false;
    uint32_t numElements = vector.getNumElements();
    if (numElements < 2 || numElements > 4)
      return false;
    Type elem = vector.getElementType();
    return elem.isF32() || elem.isInteger(32) || elem.isF16() ||
           elem.isInteger(16) || elem.isF64() || elem.isInteger(64) ||
           elem.isInteger(8) || elem.isInteger(1);
  }
  if (type.isF32() || type.isInteger(32) || type.isF16() ||
      type.isInteger(16) || type.isF64() || type.isInteger(64) ||
      type.isInteger(8) || type.isInteger(1))
    return true;

  // Memoization: if already validated as legal, return true immediately
  if (visited.contains(type))
    return true;

  // Cycle detection: if type is an active ancestor on the current stack, reject
  if (!visiting.insert(type).second)
    return false;

  bool valid = false;
  if (auto llvmStruct = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (!llvmStruct.isOpaque()) {
      valid = llvm::all_of(llvmStruct.getBody(), [&](Type member) {
        return isValidPointeeTypeImpl(member, visiting, visited);
      });
    }
  } else if (auto spirvStruct = dyn_cast<spirv::StructType>(type)) {
    valid = llvm::all_of(spirvStruct.getElementTypes(), [&](Type member) {
      return isValidPointeeTypeImpl(member, visiting, visited);
    });
  } else if (auto tuple = dyn_cast<TupleType>(type)) {
    valid = llvm::all_of(tuple.getTypes(), [&](Type member) {
      return isValidPointeeTypeImpl(member, visiting, visited);
    });
  }

  visiting.erase(type);
  if (valid)
    visited.insert(type);
  return valid;
}

bool mlir::gfx::isValidPointeeType(Type type) {
  llvm::SmallPtrSet<Type, 8> visiting;
  llvm::SmallPtrSet<Type, 16> visited;
  return isValidPointeeTypeImpl(type, visiting, visited);
}

Type PtrType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type elementType;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(elementType))
    return {};

  if (parser.parseComma())
    return {};

  std::string storageKeyword;
  llvm::SMLoc storageLoc = parser.getCurrentLocation();
  if (failed(parser.parseKeywordOrString(&storageKeyword)))
    return {};
  auto maybeStorage = symbolizeStorageClass(storageKeyword);
  if (!maybeStorage) {
    parser.emitError(storageLoc, "expected ::mlir::gfx::StorageClass to be one of: ")
        << "global, workgroup, private, push_constant";
    return {};
  }

  uint32_t alignment = 0;
  if (succeeded(parser.parseOptionalComma())) {
    llvm::SMLoc alignLoc = parser.getCurrentLocation();
    if (parser.parseInteger(alignment))
      return {};
    if (!llvm::isPowerOf2_32(alignment)) {
      parser.emitError(alignLoc, "pointer alignment must be a power of two, got ")
          << alignment;
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return parser.getChecked<PtrType>(loc, parser.getContext(), elementType,
                                    *maybeStorage, alignment);
}

void PtrType::print(AsmPrinter &printer) const {
  printer << "<" << getElementType() << ", "
          << stringifyStorageClass(getStorageClass());
  if (getAlignment() != 0)
    printer << ", " << getAlignment();
  printer << ">";
}

uint32_t PtrType::getEffectiveAlignment() const {
  uint32_t explicitAlign = getAlignment();
  return explicitAlign != 0 ? explicitAlign : getNaturalAlignment(getElementType());
}

LogicalResult PtrType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType, StorageClass storageClass,
                              uint32_t alignment) {
  if (!isValidPointeeType(elementType))
    return emitError() << "pointer element type must be a valid shader data type, got "
                       << elementType;

  if (alignment != 0) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError() << "pointer alignment must be a power of two, got " << alignment;
    uint32_t natural = getNaturalAlignment(elementType);
    if (natural > 0 && alignment < natural)
      return emitError() << "pointer alignment (" << alignment
                         << ") is smaller than the natural alignment of pointee type ("
                         << natural << ")";
  }
  return success();
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
    auto stageAttr = dyn_cast<StageAttr>(attribute.getValue());
    if (!function || !stageAttr)
      return op->emitOpError(
          "gfx.stage requires a typed stage attribute on func.func");
    if (stageAttr.getValue() == Stage::Compute) {
      if (!function.getFunctionType().getResults().empty())
        return op->emitOpError("compute entry points must not have return values");
      for (Type input : function.getFunctionType().getInputs()) {
        if (!isa<PtrType>(input))
          return op->emitOpError(
              "compute entry point arguments must be device pointers");
      }
    } else {
      for (Type input : function.getFunctionType().getInputs()) {
        if (auto ptr = dyn_cast<PtrType>(input)) {
          if (ptr.getStorageClass() == StorageClass::Workgroup)
            return op->emitOpError(
                "workgroup pointers are only allowed in compute entry points");
        }
      }
      if (!function.getFunctionType().getInputs().empty() ||
          !function.getFunctionType().getResults().empty())
        return op->emitOpError("entry points use gfx interface operations, not "
                               "host function arguments/results");
    }
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
    if (!isa<ModuleOp>(op))
      return op->emitOpError("gfx.parameters requires a typed parameter block "
                             "on the shader module");
    if (auto block = dyn_cast<ParameterBlockAttr>(attribute.getValue())) {
      // Valid single block (ParameterBlockAttr::verify already validated size <= 256)
    } else if (auto array = dyn_cast<ArrayAttr>(attribute.getValue())) {
      if (array.empty())
        return op->emitOpError("gfx.parameters requires typed parameter blocks "
                               "on the shader module");
      llvm::StringSet<> blockNames;
      uint32_t cumulativeSize = 0;
      for (Attribute elem : array) {
        auto block = dyn_cast<ParameterBlockAttr>(elem);
        if (!block)
          return op->emitOpError("gfx.parameters requires typed parameter blocks "
                                 "on the shader module");
        if (!blockNames.insert(block.getName()).second)
          return op->emitOpError("duplicate parameter block name: '")
                 << block.getName() << "'";
        auto layout = parameterLayout(block);
        cumulativeSize += (layout.size + 15) / 16 * 16;
      }
      if (cumulativeSize > 256)
        return op->emitOpError(
            "read-only parameter blocks currently fit within 256 bytes");
    } else {
      return op->emitOpError("gfx.parameters requires a typed parameter block "
                             "on the shader module");
    }
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

std::optional<BufferInterfaceElement>
mlir::gfx::getArgumentBufferBinding(func::FuncOp function, unsigned argIndex) {
  if (!function || argIndex >= function.getNumArguments())
    return std::nullopt;
  auto bindingAttr =
      function.getArgAttrOfType<IntegerAttr>(argIndex, "gfx.binding");
  if (!bindingAttr)
    return std::nullopt;
  auto ptrType = dyn_cast<PtrType>(function.getArgument(argIndex).getType());
  if (!ptrType || ptrType.getStorageClass() != StorageClass::Global)
    return std::nullopt;
  int64_t rawBinding = bindingAttr.getInt();
  if (rawBinding < 0)
    return std::nullopt;
  uint32_t set = 0;
  if (auto setAttr =
          function.getArgAttrOfType<IntegerAttr>(argIndex, "gfx.set")) {
    if (setAttr.getInt() < 0)
      return std::nullopt;
    set = static_cast<uint32_t>(setAttr.getInt());
  }
  AccessMode access = AccessMode::ReadWrite;
  if (auto bufAccess = function.getArgAttrOfType<BufferAccessAttr>(
          argIndex, "gfx.buffer_access")) {
    access = bufAccess.getValue();
  } else if (auto accessMode = function.getArgAttrOfType<AccessModeAttr>(
                 argIndex, "gfx.access_mode")) {
    access = accessMode.getValue();
  } else if (auto altAccessMode = function.getArgAttrOfType<AccessModeAttr>(
                 argIndex, "gfx.buffer_access")) {
    access = altAccessMode.getValue();
  }
  std::string name;
  if (auto nameAttr =
          function.getArgAttrOfType<StringAttr>(argIndex, "gfx.name")) {
    name = nameAttr.getValue().str();
  } else {
    name = "buffer_" + std::to_string(rawBinding);
  }
  return BufferInterfaceElement{name, set, static_cast<uint32_t>(rawBinding),
                                access, ptrType.getElementType()};
}

LogicalResult GfxDialect::verifyRegionArgAttribute(Operation *op,
                                                   unsigned regionIndex,
                                                   unsigned argIndex,
                                                   NamedAttribute attribute) {
  auto function = dyn_cast<func::FuncOp>(op);
  if (!function || regionIndex != 0 || argIndex >= function.getNumArguments())
    return op->emitOpError(
        "gfx argument attributes are only valid on func.func arguments");

  StringRef attrName = attribute.getName().getValue();
  auto ptrType = dyn_cast<PtrType>(function.getArgument(argIndex).getType());
  if (!ptrType || ptrType.getStorageClass() != StorageClass::Global) {
    return op->emitOpError("argument attribute '")
           << attrName
           << "' requires a device pointer in 'global' storage class";
  }

  if (attrName == "gfx.binding") {
    auto intAttr = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!intAttr || !intAttr.getType().isInteger(32) || intAttr.getInt() < 0)
      return op->emitOpError(
          "gfx.binding requires a non-negative 32-bit integer attribute");
    return success();
  }
  if (attrName == "gfx.set") {
    auto intAttr = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!intAttr || !intAttr.getType().isInteger(32) || intAttr.getInt() < 0)
      return op->emitOpError(
          "gfx.set requires a non-negative 32-bit integer attribute");
    return success();
  }
  if (attrName == "gfx.buffer_access" || attrName == "gfx.access_mode") {
    if (!isa<BufferAccessAttr, AccessModeAttr>(attribute.getValue()))
      return op->emitOpError("'")
             << attrName
             << "' requires a #gfx.buffer_access or #gfx.access_mode attribute";
    return success();
  }
  if (attrName == "gfx.name") {
    auto strAttr = dyn_cast<StringAttr>(attribute.getValue());
    if (!strAttr || strAttr.getValue().empty())
      return op->emitOpError("gfx.name requires a non-empty string attribute");
    return success();
  }

  return op->emitOpError("unknown gfx argument attribute: ")
         << attribute.getName();
}

LogicalResult mlir::gfx::getStd430Layout(Type type, uint32_t &size,
                                         uint32_t &alignment) {
  if (type.isF16()) {
    size = 2;
    alignment = 2;
    return success();
  }
  if (type.isF32()) {
    size = 4;
    alignment = 4;
    return success();
  }
  if (type.isF64()) {
    size = 8;
    alignment = 8;
    return success();
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 1) {
      size = 1;
      alignment = 1;
      return success();
    }
    if (width == 8) {
      size = 1;
      alignment = 1;
      return success();
    }
    if (width == 16) {
      size = 2;
      alignment = 2;
      return success();
    }
    if (width == 32) {
      size = 4;
      alignment = 4;
      return success();
    }
    if (width == 64) {
      size = 8;
      alignment = 8;
      return success();
    }
    return failure();
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1)
      return failure();
    int64_t count = vecType.getNumElements();
    if (count < 2 || count > 4)
      return failure();
    uint32_t elemSize = 0, elemAlign = 0;
    if (failed(getStd430Layout(vecType.getElementType(), elemSize, elemAlign)))
      return failure();
    if (count == 2) {
      size = 2 * elemSize;
      alignment = 2 * elemAlign;
    } else if (count == 3) {
      size = 3 * elemSize;
      alignment = 4 * elemAlign;
    } else {
      size = 4 * elemSize;
      alignment = 4 * elemAlign;
    }
    return success();
  }
  if (auto arrType = dyn_cast<ArrayType>(type)) {
    uint32_t elemSize = 0, elemAlign = 0;
    if (failed(getStd430Layout(arrType.getElementType(), elemSize, elemAlign)))
      return failure();
    uint32_t stride = (elemSize + elemAlign - 1) / elemAlign * elemAlign;
    size = arrType.getCount() * stride;
    alignment = elemAlign;
    return success();
  }
  if (auto structType = dyn_cast<StructType>(type)) {
    uint32_t currentOffset = 0;
    uint32_t maxAlign = 1;
    for (auto attr : structType.getFields()) {
      auto field = cast<ParameterFieldAttr>(attr);
      uint32_t mSize = 0, mAlign = 0;
      if (failed(getStd430Layout(field.getType(), mSize, mAlign)))
        return failure();
      maxAlign = std::max(maxAlign, mAlign);
      currentOffset = (currentOffset + mAlign - 1) / mAlign * mAlign + mSize;
    }
    alignment = maxAlign;
    size = (currentOffset + maxAlign - 1) / maxAlign * maxAlign;
    return success();
  }
  if (auto tupleType = dyn_cast<TupleType>(type)) {
    uint32_t currentOffset = 0;
    uint32_t maxAlign = 1;
    for (Type elem : tupleType.getTypes()) {
      uint32_t mSize = 0, mAlign = 0;
      if (failed(getStd430Layout(elem, mSize, mAlign)))
        return failure();
      maxAlign = std::max(maxAlign, mAlign);
      currentOffset = (currentOffset + mAlign - 1) / mAlign * mAlign + mSize;
    }
    alignment = maxAlign;
    size = (currentOffset + maxAlign - 1) / maxAlign * maxAlign;
    return success();
  }
  return failure();
}

LogicalResult
ParameterFieldAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
                           StringRef name, Type type) {
  if (!isShaderSymbolName(name))
    return error() << "parameter field requires a valid name";
  uint32_t size = 0, alignment = 0;
  if (failed(getStd430Layout(type, size, alignment)))
    return error() << "parameter fields require a supported std430 scalar, vector, array, or struct type";
  return success();
}

std::vector<ParameterFieldLayout> mlir::gfx::layoutStructMembers(StructType type) {
  std::vector<ParameterFieldLayout> members;
  uint32_t currentOffset = 0;
  for (auto attr : type.getFields()) {
    auto field = cast<ParameterFieldAttr>(attr);
    ParameterFieldLayout item;
    item.field = field;
    item.name = field.getName().str();
    item.type = field.getType();
    if (auto arr = dyn_cast<ArrayType>(field.getType())) {
      item.count = arr.getCount();
      uint32_t elemSize = 0, elemAlign = 0;
      if (succeeded(getStd430Layout(arr.getElementType(), elemSize, elemAlign))) {
        item.stride = (elemSize + elemAlign - 1) / elemAlign * elemAlign;
      }
      if (auto elemSt = dyn_cast<StructType>(arr.getElementType())) {
        item.nestedFields = layoutStructMembers(elemSt);
      }
    } else if (auto subSt = dyn_cast<StructType>(field.getType())) {
      item.nestedFields = layoutStructMembers(subSt);
    }
    uint32_t fSize = 0, fAlign = 0;
    if (succeeded(getStd430Layout(field.getType(), fSize, fAlign))) {
      item.size = fSize;
      item.alignment = fAlign;
      uint32_t alignedOffset = (currentOffset + fAlign - 1) / fAlign * fAlign;
      item.offset = alignedOffset;
      if (!members.empty()) {
        members.back().paddingAfter =
            alignedOffset - (members.back().offset + members.back().size);
      }
      currentOffset = alignedOffset + fSize;
      members.push_back(item);
    }
  }
  uint32_t stSize = 0, stAlign = 0;
  if (succeeded(getStd430Layout(type, stSize, stAlign)) && !members.empty()) {
    members.back().paddingAfter =
        stSize - (members.back().offset + members.back().size);
  }
  return members;
}

static ParameterLayout layoutFields(ArrayAttr fields) {
  ParameterLayout layout;
  uint32_t currentOffset = 0;
  uint32_t maxAlign = 1;
  for (auto attribute : fields) {
    auto field = cast<ParameterFieldAttr>(attribute);
    ParameterFieldLayout item;
    item.field = field;
    item.name = field.getName().str();
    item.type = field.getType();
    if (auto arr = dyn_cast<ArrayType>(field.getType())) {
      item.count = arr.getCount();
      uint32_t elemSize = 0, elemAlign = 0;
      if (succeeded(getStd430Layout(arr.getElementType(), elemSize, elemAlign))) {
        item.stride = (elemSize + elemAlign - 1) / elemAlign * elemAlign;
      }
      if (auto elemSt = dyn_cast<StructType>(arr.getElementType())) {
        item.nestedFields = layoutStructMembers(elemSt);
      }
    } else if (auto st = dyn_cast<StructType>(field.getType())) {
      item.nestedFields = layoutStructMembers(st);
    }
    uint32_t fSize = 0, fAlign = 0;
    if (succeeded(getStd430Layout(field.getType(), fSize, fAlign))) {
      item.size = fSize;
      item.alignment = fAlign;
      maxAlign = std::max(maxAlign, fAlign);
      uint32_t alignedOffset = (currentOffset + fAlign - 1) / fAlign * fAlign;
      item.offset = alignedOffset;
      if (!layout.fields.empty()) {
        layout.fields.back().paddingAfter =
            alignedOffset - (layout.fields.back().offset + layout.fields.back().size);
      }
      currentOffset = alignedOffset + fSize;
      layout.fields.push_back(item);
    }
  }
  layout.alignment = maxAlign;
  uint32_t totalSize = (currentOffset + maxAlign - 1) / maxAlign * maxAlign;
  layout.size = totalSize;
  if (!layout.fields.empty()) {
    layout.fields.back().paddingAfter =
        totalSize - (layout.fields.back().offset + layout.fields.back().size);
  }
  return layout;
}

ParameterLayout mlir::gfx::parameterLayout(ParameterBlockAttr block) {
  return layoutFields(block.getFields());
}

SmallVector<ParameterBlockAttr>
mlir::gfx::getModuleParameterBlocks(ModuleOp module) {
  SmallVector<ParameterBlockAttr> result;
  if (!module)
    return result;
  if (auto single =
          module->getAttrOfType<ParameterBlockAttr>("gfx.parameters")) {
    result.push_back(single);
  } else if (auto array = module->getAttrOfType<ArrayAttr>("gfx.parameters")) {
    for (auto attr : array) {
      if (auto b = dyn_cast<ParameterBlockAttr>(attr))
        result.push_back(b);
    }
  }
  return result;
}

LogicalResult
ParameterBlockAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
                           StringRef name, ArrayAttr fields) {
  if (!isShaderSymbolName(name) || fields.empty() || fields.size() > 64)
    return error() << "parameter block requires a valid name and 1-64 fields";
  llvm::StringSet<> names;
  for (auto attribute : fields) {
    auto field = dyn_cast<ParameterFieldAttr>(attribute);
    if (!field || !names.insert(field.getName()).second)
      return error() << "parameter fields must be typed and uniquely named";
    if (failed(ParameterFieldAttr::verify(error, field.getName(),
                                          field.getType())))
      return failure();
  }
  if (layoutFields(fields).size > 256)
    return error()
           << "read-only parameter blocks currently fit within 256 bytes";
  return success();
}

static Type toSignlessParameterValueType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (!intType.isSignless())
      return IntegerType::get(type.getContext(), intType.getWidth());
    return type;
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    Type elem = toSignlessParameterValueType(vecType.getElementType());
    if (elem != vecType.getElementType())
      return VectorType::get(vecType.getShape(), elem);
  }
  return type;
}

LogicalResult ParameterOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto blocks = getModuleParameterBlocks(module);
  if (!(*this)->getParentOfType<func::FuncOp>() || blocks.empty())
    return emitOpError("requires a matching declared parameter block");
  ParameterBlockAttr matchedBlock;
  for (auto b : blocks) {
    if (b.getName() == getBlock()) {
      matchedBlock = b;
      break;
    }
  }
  if (!matchedBlock)
    return emitOpError("requires a matching declared parameter block");
  for (auto attribute : matchedBlock.getFields()) {
    auto field = cast<ParameterFieldAttr>(attribute);
    if (field.getName() == getField()) {
      Type fType = field.getType();
      Type valType = getValue().getType();
      if (fType == valType || toSignlessParameterValueType(fType) == valType)
        return success();
      if (auto fInt = dyn_cast<IntegerType>(fType)) {
        if (auto valInt = dyn_cast<IntegerType>(valType)) {
          if (fInt.getWidth() == valInt.getWidth())
            return success();
        }
      }
      return emitOpError("parameter result type does not match its field");
    }
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

Attribute AccessModeAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  std::string keyword;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeywordOrString(&keyword))
    return {};
  if (parser.parseGreater())
    return {};
  auto maybe = symbolizeAccessMode(keyword);
  if (!maybe) {
    parser.emitError(loc, "expected ::mlir::gfx::AccessMode to be one of: read_only, write_only, read_write");
    return {};
  }
  return AccessModeAttr::get(parser.getContext(), *maybe, type);
}

void AccessModeAttr::print(AsmPrinter &printer) const {
  printer << "<" << stringifyAccessMode(getValue()) << ">";
}

ParseResult BufferBindingOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr setAttr;
  IntegerAttr bindingAttr;
  StringAttr nameAttr;
  AccessModeAttr accessAttr;
  PtrType ptrType;

  // Support formal keyword syntax:
  //   gfx.buffer_binding @name [set(S)] binding(B) access(mode) : !gfx.ptr<T, global>
  if (succeeded(parser.parseOptionalSymbolName(nameAttr))) {
    int32_t setVal = 0;
    if (succeeded(parser.parseOptionalKeyword("set"))) {
      if (parser.parseLParen() || parser.parseInteger(setVal) ||
          parser.parseRParen())
        return failure();
    }
    int32_t bindingVal = 0;
    if (parser.parseKeyword("binding") || parser.parseLParen() ||
        parser.parseInteger(bindingVal) || parser.parseRParen())
      return failure();

    std::string modeKeyword;
    llvm::SMLoc modeLoc = parser.getCurrentLocation();
    if (parser.parseKeyword("access") || parser.parseLParen() ||
        parser.parseKeywordOrString(&modeKeyword) || parser.parseRParen())
      return failure();
    auto maybeMode = symbolizeAccessMode(modeKeyword);
    if (!maybeMode) {
      return parser.emitError(
          modeLoc,
          "expected ::mlir::gfx::AccessMode to be one of: read_only, write_only, read_write");
    }
    setAttr = parser.getBuilder().getI32IntegerAttr(setVal);
    bindingAttr = parser.getBuilder().getI32IntegerAttr(bindingVal);
    accessAttr =
        AccessModeAttr::get(parser.getContext(), *maybeMode, Type{});

    if (parser.parseColon())
      return failure();
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    Type rawType;
    if (parser.parseType(rawType))
      return failure();
    if (!isa<PtrType>(rawType))
      return parser.emitError(typeLoc, "result #0 must be !gfx.ptr, but got '")
             << rawType << "'";
    ptrType = cast<PtrType>(rawType);

    auto &props = result.getOrAddProperties<BufferBindingOp::Properties>();
    props.set = setAttr;
    props.binding = bindingAttr;
    props.name = nameAttr;
    props.access = accessAttr;
    result.addTypes(ptrType);
    return success();
  }

  if (parser.parseCustomAttributeWithFallback(setAttr, parser.getBuilder().getIntegerType(32)) ||
      parser.parseComma() ||
      parser.parseCustomAttributeWithFallback(bindingAttr, parser.getBuilder().getIntegerType(32)) ||
      parser.parseComma() ||
      parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<NoneType>()) ||
      parser.parseComma())
    return failure();

  llvm::SMLoc accessLoc = parser.getCurrentLocation();
  Attribute rawAccessAttr;
  if (parser.parseAttribute(rawAccessAttr))
    return failure();
  if (auto am = dyn_cast<AccessModeAttr>(rawAccessAttr)) {
    accessAttr = am;
  } else if (auto ba = dyn_cast<BufferAccessAttr>(rawAccessAttr)) {
    accessAttr =
        AccessModeAttr::get(parser.getContext(), ba.getValue(), Type{});
  } else {
    return parser.emitError(
        accessLoc,
        "expected #gfx.access_mode or #gfx.buffer_access attribute");
  }

  if (setAttr)
    result.getOrAddProperties<BufferBindingOp::Properties>().set = setAttr;
  if (bindingAttr)
    result.getOrAddProperties<BufferBindingOp::Properties>().binding = bindingAttr;
  if (nameAttr)
    result.getOrAddProperties<BufferBindingOp::Properties>().name = nameAttr;
  if (accessAttr)
    result.getOrAddProperties<BufferBindingOp::Properties>().access = accessAttr;

  if (accessAttr && accessAttr.getType()) {
    if (!isa<PtrType>(accessAttr.getType()))
      return parser.emitError(accessLoc, "result #0 must be !gfx.ptr, but got '") << accessAttr.getType() << "'";
    ptrType = cast<PtrType>(accessAttr.getType());
  } else {
    if (parser.parseColon())
      return failure();
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    Type rawType;
    if (parser.parseType(rawType))
      return failure();
    if (!isa<PtrType>(rawType))
      return parser.emitError(typeLoc, "result #0 must be !gfx.ptr, but got '") << rawType << "'";
    ptrType = cast<PtrType>(rawType);
  }

  result.addTypes(ptrType);
  return success();
}

void BufferBindingOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getSet() << ", " << getBinding() << ", " << getNameAttr() << ", "
          << "#gfx.access_mode<" << stringifyAccessMode(getAccess().getValue()) << "> : "
          << getResult().getType();
}

LogicalResult BufferBindingOp::verify() {
  if (getSetAttr().getInt() < 0)
    return emitOpError("buffer binding set index must be non-negative, got ") << getSetAttr().getInt();
  if (getBindingAttr().getInt() < 0)
    return emitOpError("buffer binding slot index must be non-negative, got ") << getBindingAttr().getInt();
  if (getName().empty())
    return emitOpError("buffer name must not be empty");

  auto ptrType = cast<PtrType>(getResult().getType());
  if (ptrType.getStorageClass() != StorageClass::Global)
    return emitOpError("buffer binding must produce a pointer with 'global' storage class, got '")
           << stringifyStorageClass(ptrType.getStorageClass()) << "'";

  return success();
}

LogicalResult PtrOffsetOp::verify() {
  auto ptrType = cast<PtrType>(getPtr().getType());
  auto resType = cast<PtrType>(getResult().getType());

  if (resType.getElementType() != ptrType.getElementType())
    return emitOpError("result pointer element type ")
           << resType.getElementType()
           << " does not match source pointer element type "
           << ptrType.getElementType();

  if (resType.getStorageClass() != ptrType.getStorageClass())
    return emitOpError("result pointer storage class '")
           << stringifyStorageClass(resType.getStorageClass())
           << "' does not match source pointer storage class '"
           << stringifyStorageClass(ptrType.getStorageClass()) << "'";

  return success();
}

static Value getRootPointerValue(Value ptr) {
  while (auto offsetOp = ptr.getDefiningOp<PtrOffsetOp>())
    ptr = offsetOp.getPtr();
  return ptr;
}

static std::optional<AccessMode> getBlockArgumentAccessMode(Value rootVal,
                                                            std::string &label) {
  auto blockArg = dyn_cast<BlockArgument>(rootVal);
  if (!blockArg)
    return std::nullopt;
  Operation *parentOp = blockArg.getOwner() ? blockArg.getOwner()->getParentOp()
                                            : nullptr;
  auto funcOp = dyn_cast_or_null<func::FuncOp>(parentOp);
  if (!funcOp || blockArg.getOwner() != &funcOp.getBody().front())
    return std::nullopt;
  unsigned argIdx = blockArg.getArgNumber();
  if (auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(argIdx, "gfx.name"))
    label = nameAttr.getValue().str();
  else
    label = "arg" + std::to_string(argIdx);

  if (auto ba = funcOp.getArgAttrOfType<BufferAccessAttr>(argIdx, "gfx.buffer_access"))
    return ba.getValue();
  if (auto am = funcOp.getArgAttrOfType<AccessModeAttr>(argIdx, "gfx.access_mode"))
    return am.getValue();
  if (auto am2 = funcOp.getArgAttrOfType<AccessModeAttr>(argIdx, "gfx.buffer_access"))
    return am2.getValue();
  return std::nullopt;
}

LogicalResult LoadOp::verify() {
  auto ptrType = cast<PtrType>(getPtr().getType());

  if (getResult().getType() != ptrType.getElementType())
    return emitOpError("load result type ")
           << getResult().getType()
           << " does not match pointer element type "
           << ptrType.getElementType();

  if (getAlignment().has_value()) {
    int32_t align = *getAlignment();
    if (align <= 0 || !llvm::isPowerOf2_32(align))
      return emitOpError("load alignment must be a power of two, got ") << align;
    uint32_t natural = getNaturalAlignment(ptrType.getElementType());
    if (natural > 0 && static_cast<uint32_t>(align) < natural)
      return emitOpError("load alignment (") << align
             << ") is smaller than natural alignment of pointee type (" << natural << ")";
  }

  Value rootVal = getRootPointerValue(getPtr());
  if (auto binding = rootVal.getDefiningOp<BufferBindingOp>()) {
    if (binding.getAccess() == AccessMode::WriteOnly)
      return emitOpError("cannot load from write-only buffer binding '")
             << binding.getName() << "'";
  } else {
    std::string argLabel;
    if (auto mode = getBlockArgumentAccessMode(rootVal, argLabel)) {
      if (*mode == AccessMode::WriteOnly)
        return emitOpError("cannot load from write-only buffer argument '")
               << argLabel << "'";
    }
  }

  return success();
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand valueOperand;
  OpAsmParser::UnresolvedOperand ptrOperand;
  IntegerAttr alignmentAttr;
  Type valueType;
  Type ptrType;

  if (parser.parseOperand(valueOperand) ||
      parser.parseComma() ||
      parser.parseOperand(ptrOperand))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("alignment") ||
        parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(alignmentAttr, parser.getBuilder().getIntegerType(32)))
      return failure();
    if (alignmentAttr)
      result.getOrAddProperties<StoreOp::Properties>().alignment = alignmentAttr;
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon())
    return failure();

  bool hasParens = succeeded(parser.parseOptionalLParen());
  if (parser.parseType(valueType) ||
      parser.parseComma() ||
      parser.parseType(ptrType))
    return failure();
  if (hasParens && parser.parseRParen())
    return failure();

  if (parser.resolveOperand(valueOperand, valueType, result.operands) ||
      parser.resolveOperand(ptrOperand, ptrType, result.operands))
    return failure();

  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getValue() << ", " << getPtr();
  if (getAlignment().has_value())
    printer << ", alignment = " << *getAlignment();
  printer.printOptionalAttrDict((*this)->getDiscardableAttrDictionary().getValue());
  printer << " : " << getValue().getType() << ", " << getPtr().getType();
}

LogicalResult StoreOp::verify() {
  auto ptrType = cast<PtrType>(getPtr().getType());

  if (getValue().getType() != ptrType.getElementType())
    return emitOpError("stored value type ")
           << getValue().getType()
           << " does not match pointer element type "
           << ptrType.getElementType();

  if (ptrType.getStorageClass() == StorageClass::PushConstant)
    return emitOpError("cannot store into read-only storage class 'push_constant'");

  if (getAlignment().has_value()) {
    int32_t align = *getAlignment();
    if (align <= 0 || !llvm::isPowerOf2_32(align))
      return emitOpError("store alignment must be a power of two, got ") << align;
    uint32_t natural = getNaturalAlignment(ptrType.getElementType());
    if (natural > 0 && static_cast<uint32_t>(align) < natural)
      return emitOpError("store alignment (") << align
             << ") is smaller than natural alignment of pointee type (" << natural << ")";
  }

  Value rootVal = getRootPointerValue(getPtr());
  if (auto binding = rootVal.getDefiningOp<BufferBindingOp>()) {
    if (binding.getAccess() == AccessMode::ReadOnly)
      return emitOpError("cannot store to read-only buffer binding '")
             << binding.getName() << "'";
  } else {
    std::string argLabel;
    if (auto mode = getBlockArgumentAccessMode(rootVal, argLabel)) {
      if (*mode == AccessMode::ReadOnly)
        return emitOpError("cannot store to read-only buffer argument '")
               << argLabel << "'";
    }
  }

  return success();
}

LogicalResult SharedAllocOp::verify() {
  if (!(*this)->getParentOfType<func::FuncOp>())
    return emitOpError("requires an enclosing function");
  if (auto stage = stageOf(*this)) {
    if (stage.getValue() != Stage::Compute)
      return emitOpError(
          "shared memory allocations are only allowed in compute shaders");
  }
  int64_t count = getCountAttr().getInt();
  if (count <= 0)
    return emitOpError("shared memory element count must be positive, got ")
           << count;
  if (getName().has_value() && !isShaderSymbolName(*getName()))
    return emitOpError("shared memory name must be a valid identifier");

  auto ptrType = cast<PtrType>(getResult().getType());
  if (ptrType.getStorageClass() != StorageClass::Workgroup)
    return emitOpError(
               "shared_alloc must produce a pointer with 'workgroup' storage class, got '")
           << stringifyStorageClass(ptrType.getStorageClass()) << "'";

  uint32_t elemSize = 0, elemAlign = 0;
  if (failed(getStd430Layout(ptrType.getElementType(), elemSize, elemAlign)) ||
      elemSize == 0)
    return emitOpError("shared_alloc element type ")
           << ptrType.getElementType() << " does not have a supported layout";

  return success();
}

static LogicalResult verifyUniformBarrierPlacement(BarrierOp barrier,
                                                   func::FuncOp func) {
  llvm::DenseSet<Value> divergent;
  auto markDivergent = [&](Value v) -> bool {
    return v && divergent.insert(v).second;
  };

  func.walk([&](Operation *op) {
    if (auto builtin = dyn_cast<BuiltinOp>(op)) {
      if (builtin.getKind() == Builtin::LocalInvocationId ||
          builtin.getKind() == Builtin::GlobalInvocationId) {
        markDivergent(builtin.getValue());
      }
    } else if (auto rmw = dyn_cast<AtomicRMWOp>(op)) {
      markDivergent(rmw.getResult());
    } else if (auto cas = dyn_cast<AtomicCASOp>(op)) {
      markDivergent(cas.getResult());
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](Operation *op) {
      if (isa<BuiltinOp, ParameterOp, SharedAllocOp, BufferBindingOp,
              arith::ConstantOp>(op))
        return;

      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        bool condDiv = divergent.contains(ifOp.getCondition());
        auto thenYield =
            cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
        scf::YieldOp elseYield;
        if (!ifOp.getElseRegion().empty()) {
          elseYield =
              cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
        }
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
          bool resDiv = condDiv || divergent.contains(thenYield.getOperand(i)) ||
                        (elseYield && divergent.contains(elseYield.getOperand(i)));
          if (resDiv)
            changed |= markDivergent(ifOp.getResult(i));
        }
        return;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        bool boundsDiv = divergent.contains(forOp.getLowerBound()) ||
                         divergent.contains(forOp.getUpperBound()) ||
                         divergent.contains(forOp.getStep());
        if (boundsDiv)
          changed |= markDivergent(forOp.getInductionVar());
        auto yieldOp =
            cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
          bool iterDiv = boundsDiv ||
                         divergent.contains(forOp.getInitArgs()[i]) ||
                         divergent.contains(yieldOp.getOperand(i));
          if (iterDiv) {
            changed |= markDivergent(forOp.getRegionIterArgs()[i]);
            changed |= markDivergent(forOp.getResult(i));
          }
        }
        return;
      }

      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        auto condOp = cast<scf::ConditionOp>(
            whileOp.getBefore().front().getTerminator());
        auto yieldOp =
            cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
        bool condDiv = divergent.contains(condOp.getCondition());
        for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
          bool argDiv =
              condDiv || divergent.contains(whileOp.getInits()[i]) ||
              (i < yieldOp.getNumOperands() &&
               divergent.contains(yieldOp.getOperand(i)));
          if (argDiv && i < whileOp.getBeforeArguments().size())
            changed |= markDivergent(whileOp.getBeforeArguments()[i]);
        }
        for (unsigned i = 0; i < condOp.getArgs().size(); ++i) {
          bool afterDiv = condDiv || divergent.contains(condOp.getArgs()[i]);
          if (afterDiv) {
            if (i < whileOp.getAfterArguments().size())
              changed |= markDivergent(whileOp.getAfterArguments()[i]);
            if (i < whileOp.getNumResults())
              changed |= markDivergent(whileOp.getResult(i));
          }
        }
        return;
      }

      bool anyOperandDiv = llvm::any_of(op->getOperands(), [&](Value operand) {
        return divergent.contains(operand);
      });
      if (anyOperandDiv) {
        for (Value res : op->getResults())
          changed |= markDivergent(res);
      }
    });
  }

  for (Operation *parent = barrier->getParentOp();
       parent && parent != func.getOperation();
       parent = parent->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      if (divergent.contains(ifOp.getCondition()))
        return barrier.emitOpError(
            "barrier cannot be placed inside thread-divergent control flow");
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (divergent.contains(forOp.getLowerBound()) ||
          divergent.contains(forOp.getUpperBound()) ||
          divergent.contains(forOp.getStep())) {
        return barrier.emitOpError(
            "barrier cannot be placed inside thread-divergent control flow");
      }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      auto condOp =
          cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
      if (divergent.contains(condOp.getCondition())) {
        return barrier.emitOpError(
            "barrier cannot be placed inside thread-divergent control flow");
      }
    }
  }

  Block *block = barrier->getBlock();
  if (block && block != &func.getBody().front() &&
      llvm::isa<func::FuncOp>(block->getParentOp())) {
    for (Block *pred : block->getPredecessors()) {
      if (auto condBr = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (divergent.contains(condBr.getCondition())) {
          return barrier.emitOpError(
              "barrier cannot be placed inside thread-divergent control flow");
        }
      }
    }
  }

  return success();
}

LogicalResult BarrierOp::verify() {
  auto func = (*this)->getParentOfType<func::FuncOp>();
  if (!func)
    return emitOpError("requires an enclosing function");
  if (auto stage = stageOf(*this)) {
    if (stage.getValue() != Stage::Compute)
      return emitOpError("barriers are only allowed in compute shaders");
  }
  return verifyUniformBarrierPlacement(*this, func);
}

ParseResult BarrierOp::parse(OpAsmParser &parser, OperationState &result) {
  MemoryScope scopeVal = MemoryScope::Workgroup;
  if (succeeded(parser.parseOptionalLess())) {
    std::string keyword;
    llvm::SMLoc kwLoc = parser.getCurrentLocation();
    if (parser.parseKeywordOrString(&keyword) || parser.parseGreater())
      return failure();
    auto maybeScope = symbolizeMemoryScope(keyword);
    if (!maybeScope) {
      return parser.emitError(
          kwLoc,
          "expected ::mlir::gfx::MemoryScope to be one of: workgroup, device, all");
    }
    scopeVal = *maybeScope;
  } else {
    MemoryScopeAttr attr;
    OptionalParseResult optRes =
        parser.parseOptionalAttribute(attr, Type{});
    if (optRes.has_value()) {
      if (failed(*optRes))
        return failure();
      scopeVal = attr.getValue();
    }
  }
  result.getOrAddProperties<BarrierOp::Properties>().scope =
      MemoryScopeAttr::get(parser.getContext(), scopeVal);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void BarrierOp::print(OpAsmPrinter &printer) {
  if (getScope() != MemoryScope::Workgroup)
    printer << " <" << stringifyMemoryScope(getScope()) << ">";
  printer.printOptionalAttrDict(
      (*this)->getDiscardableAttrDictionary().getValue());
}

static LogicalResult verifyAtomicPointerAndAccess(Operation *op, Value ptrVal) {
  auto ptrType = cast<PtrType>(ptrVal.getType());
  StorageClass sc = ptrType.getStorageClass();
  if (sc != StorageClass::Global && sc != StorageClass::Workgroup) {
    return op->emitOpError(
               "atomic operations require 'global' or 'workgroup' storage class, got '")
           << stringifyStorageClass(sc) << "'";
  }
  Type elemType = ptrType.getElementType();
  if (!elemType.isInteger(32)) {
    return op->emitOpError(
               "atomic operations require a 32-bit integer pointee type, got ")
           << elemType;
  }
  if (sc == StorageClass::Global) {
    Value rootVal = getRootPointerValue(ptrVal);
    if (auto binding = rootVal.getDefiningOp<BufferBindingOp>()) {
      if (binding.getAccess() != AccessMode::ReadWrite)
        return op->emitOpError(
                   "atomic operations require read_write buffer binding '")
               << binding.getName() << "', got '"
               << stringifyAccessMode(binding.getAccess()) << "'";
    } else {
      std::string argLabel;
      if (auto mode = getBlockArgumentAccessMode(rootVal, argLabel)) {
        if (*mode != AccessMode::ReadWrite)
          return op->emitOpError(
                     "atomic operations require read_write buffer argument '")
                 << argLabel << "', got '" << stringifyAccessMode(*mode) << "'";
      }
    }
  }
  return success();
}

LogicalResult AtomicRMWOp::verify() {
  if (failed(verifyAtomicPointerAndAccess(*this, getPtr())))
    return failure();
  auto ptrType = cast<PtrType>(getPtr().getType());
  Type elemType = ptrType.getElementType();
  if (getValue().getType() != elemType) {
    return emitOpError("atomic value operand type ")
           << getValue().getType() << " does not match pointer element type "
           << elemType;
  }
  if (getResult().getType() != elemType) {
    return emitOpError("atomic result type ")
           << getResult().getType() << " does not match pointer element type "
           << elemType;
  }
  return success();
}

LogicalResult AtomicCASOp::verify() {
  if (failed(verifyAtomicPointerAndAccess(*this, getPtr())))
    return failure();
  auto ptrType = cast<PtrType>(getPtr().getType());
  Type elemType = ptrType.getElementType();
  if (getCmp().getType() != elemType) {
    return emitOpError("atomic compare operand type ")
           << getCmp().getType() << " does not match pointer element type "
           << elemType;
  }
  if (getValue().getType() != elemType) {
    return emitOpError("atomic exchange operand type ")
           << getValue().getType() << " does not match pointer element type "
           << elemType;
  }
  if (getResult().getType() != elemType) {
    return emitOpError("atomic result type ")
           << getResult().getType() << " does not match pointer element type "
           << elemType;
  }
  return success();
}

uint64_t mlir::gfx::computeSharedMemoryBytes(ModuleOp module,
                                             func::FuncOp entry) {
  if (!entry)
    return 0;
  uint64_t allocTotal = 0;
  SmallVector<func::FuncOp, 8> pending{entry};
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (!pending.empty()) {
    func::FuncOp current = pending.pop_back_val();
    if (!visited.insert(current.getOperation()).second)
      continue;
    current.walk([&](SharedAllocOp alloc) {
      auto ptrType = cast<PtrType>(alloc.getResult().getType());
      uint32_t elemSize = 0, elemAlign = 0;
      if (succeeded(
              getStd430Layout(ptrType.getElementType(), elemSize, elemAlign)) &&
          elemSize > 0 && elemAlign > 0) {
        uint64_t stride = llvm::alignTo(elemSize, elemAlign);
        int64_t count = alloc.getCountAttr().getInt();
        if (count > 0) {
          allocTotal = llvm::alignTo(allocTotal, elemAlign);
          allocTotal += static_cast<uint64_t>(count) * stride;
        }
      }
    });
    current.walk([&](func::CallOp call) {
      if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        pending.push_back(callee);
    });
  }
  uint64_t attrBytes = 0;
  if (auto sharedAttr =
          entry->getAttrOfType<IntegerAttr>("gfx.shared_memory_bytes")) {
    if (sharedAttr.getInt() > 0)
      attrBytes = static_cast<uint64_t>(sharedAttr.getInt());
  }
  return std::max(allocTotal, attrBytes);
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
      uint64_t totalSharedBytes = computeSharedMemoryBytes(module, function);
      if (totalSharedBytes > 0) {
        if (failed(
                profile.verifySharedMemory(totalSharedBytes, function.getLoc())))
          return failure();
      }
    } else {
      SmallVector<func::FuncOp, 8> pending{function};
      llvm::SmallPtrSet<Operation *, 8> visited;
      while (!pending.empty()) {
        func::FuncOp current = pending.pop_back_val();
        if (!visited.insert(current.getOperation()).second)
          continue;
        WalkResult computeOnlyCheck = current.walk([&](Operation *op) {
          if (isa<SharedAllocOp>(op)) {
            op->emitOpError(
                "shared memory allocations are only allowed in compute shaders");
            return WalkResult::interrupt();
          }
          if (isa<BarrierOp>(op)) {
            op->emitOpError("barriers are only allowed in compute shaders");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (computeOnlyCheck.wasInterrupted())
          return failure();
        current.walk([&](func::CallOp call) {
          if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
            pending.push_back(callee);
        });
      }
    }
    for (Type argType : function.getArgumentTypes()) {
      if (auto ptr = dyn_cast<PtrType>(argType)) {
        if (ptr.getStorageClass() == StorageClass::Workgroup &&
            stage.getValue() != Stage::Compute) {
          return function.emitOpError(
              "workgroup pointers are only allowed in compute entry points");
        }
      }
    }
    WalkResult ptrCheck = function.walk([&](Operation *op) {
      auto checkType = [&](Type t) -> LogicalResult {
        if (auto ptr = dyn_cast<PtrType>(t)) {
          if (ptr.getStorageClass() == StorageClass::Workgroup &&
              stage.getValue() != Stage::Compute) {
            return op->emitOpError(
                "workgroup pointers are only allowed in compute entry points");
          }
        }
        return success();
      };
      for (Type operandType : op->getOperandTypes()) {
        if (failed(checkType(operandType)))
          return WalkResult::interrupt();
      }
      for (Type resultType : op->getResultTypes()) {
        if (failed(checkType(resultType)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (ptrCheck.wasInterrupted())
      return failure();
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
TargetProfile::verifySharedMemory(uint64_t bytes, mlir::Location loc) const {
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
