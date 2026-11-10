#include "Gfx/Gfx.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/ADT/StringMap.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::gfx;

namespace {
using Values = llvm::StringMap<Value>;
using Blocks = llvm::StringMap<Block *>;

class StructuredImporter {
public:
  explicit StructuredImporter(MLIRContext &context) : builder(&context) {}
  OwningOpRef<ModuleOp> import(const llvm::json::Object &source);

private:
  OpBuilder builder;

  Type type(StringRef name) {
    if (name == "index")
      return builder.getIndexType();
    if (name.starts_with("!") || name.starts_with("vector<")) {
      if (Type parsed = mlir::parseType(name, builder.getContext()))
        return parsed;
    }
    Type shaderType = parseShaderType(*builder.getContext(), name);
    if (shaderType)
      return shaderType;
    return mlir::parseType(name, builder.getContext());
  }

  LogicalResult error(Location location, StringRef message) {
    emitError(location, message);
    return failure();
  }

  FailureOr<Location> location(const llvm::json::Object &object,
                               Location parent) {
    if (!object.get("source_file") && !object.get("source_line") &&
        !object.get("source_column"))
      return parent;
    auto file = object.getString("source_file");
    auto line = object.getInteger("source_line");
    auto column = object.getInteger("source_column");
    if (!file || !line || *line < 1 || *line > UINT32_MAX)
      return error(parent, "source location requires a file and positive "
                           "32-bit source_line");
    if (object.get("source_column") &&
        (!column || *column < 1 || *column > UINT32_MAX))
      return error(parent, "source_column must be a positive 32-bit integer");
    return Location(FileLineColLoc::get(builder.getContext(), *file, *line,
                                        column ? *column : 1));
  }

  FailureOr<TypedAttr> literal(Type type, const llvm::json::Value &value,
                               Location loc) {
    if (type.isF32()) {
      auto number = value.getAsNumber();
      if (number && std::isfinite(*number) &&
          std::abs(*number) <= std::numeric_limits<float>::max())
        return TypedAttr(builder.getF32FloatAttr(*number));
    } else if (type.isInteger(1)) {
      if (auto boolean = value.getAsBoolean())
        return TypedAttr(builder.getBoolAttr(*boolean));
    } else if (type.isInteger(32) || type.isIndex()) {
      auto number = value.getAsInteger();
      if (number &&
          (type.isIndex() || (*number >= INT32_MIN && *number <= UINT32_MAX)))
        return TypedAttr(builder.getIntegerAttr(
            type.isIndex() ? type : builder.getIntegerType(type.getIntOrFloatBitWidth()),
            *number));
    }
    return error(loc, "attribute literal does not match its declared type");
  }

  FailureOr<Attribute> attribute(const llvm::json::Value &description,
                                 Location loc) {
    auto object = description.getAsObject();
    auto kind = object ? object->getString("kind") : std::nullopt;
    auto value = object ? object->get("value") : nullptr;
    if (!kind || !value)
      return error(loc, "attribute requires a kind and value");
    if (*kind == "string" || *kind == "symbol") {
      auto text = value->getAsString();
      if (!text || (*kind == "symbol" && !isShaderSymbolName(*text)))
        return error(loc, "string/symbol attribute requires a valid value");
      return *kind == "string" ? Attribute(builder.getStringAttr(*text))
                               : Attribute(FlatSymbolRefAttr::get(
                                     builder.getContext(), *text));
    }
    if (*kind == "literal" || *kind == "dense") {
      auto name = object->getString("type");
      auto parsedType = name ? type(*name) : Type{};
      if (!parsedType)
        return error(loc, "attribute requires a supported type");
      if (*kind == "literal") {
        auto result = literal(parsedType, *value, loc);
        if (failed(result))
          return failure();
        return Attribute(*result);
      }
      auto vector = dyn_cast<VectorType>(parsedType);
      auto elements = value->getAsArray();
      if (!vector || !elements ||
          (elements->size() != 1 &&
           elements->size() != static_cast<size_t>(vector.getNumElements())))
        return error(loc, "dense attribute requires one splat or matching "
                          "vector components");
      SmallVector<Attribute> literals;
      for (const auto &element : *elements) {
        auto result = literal(vector.getElementType(), element, loc);
        if (failed(result))
          return failure();
        literals.push_back(*result);
      }
      return Attribute(DenseElementsAttr::get(vector, literals));
    }
    if (*kind == "i32_array" || *kind == "i64_array") {
      auto elements = value->getAsArray();
      if (!elements)
        return error(loc, "integer array attribute requires an array");
      SmallVector<int64_t> numbers;
      for (const auto &element : *elements) {
        auto number = element.getAsInteger();
        if (!number || (*kind == "i32_array" &&
                        (*number < INT32_MIN || *number > INT32_MAX)))
          return error(loc, "integer array element is out of range");
        numbers.push_back(*number);
      }
      if (*kind == "i64_array")
        return Attribute(builder.getDenseI64ArrayAttr(numbers));
      SmallVector<int32_t> narrowed(numbers.begin(), numbers.end());
      return Attribute(builder.getDenseI32ArrayAttr(narrowed));
    }
    auto text = value->getAsString();
    if (text) {
      if (*kind == "access_mode")
        if (auto item = symbolizeAccessMode(*text))
          return Attribute(AccessModeAttr::get(builder.getContext(), *item, Type{}));
      if (*kind == "builtin")
        if (auto item = symbolizeBuiltin(*text))
          return Attribute(BuiltinAttr::get(builder.getContext(), *item));
      if (*kind == "interpolation")
        if (auto item = symbolizeInterpolation(*text))
          return Attribute(InterpolationAttr::get(builder.getContext(), *item));
      if (*kind == "cmpi")
        if (auto item = arith::symbolizeCmpIPredicate(*text))
          return Attribute(
              arith::CmpIPredicateAttr::get(builder.getContext(), *item));
      if (*kind == "cmpf")
        if (auto item = arith::symbolizeCmpFPredicate(*text))
          return Attribute(
              arith::CmpFPredicateAttr::get(builder.getContext(), *item));
    }
    return error(loc, "unknown attribute kind or enum value");
  }

  LogicalResult region(const llvm::json::Object &description, Region &region,
                       const Values &captures, Location parent);
  LogicalResult operation(const llvm::json::Object &description, Values &values,
                          const Blocks &blocks, Location parent);
};

LogicalResult
StructuredImporter::operation(const llvm::json::Object &description,
                              Values &values, const Blocks &blocks,
                              Location parent) {
  auto loc = location(description, parent);
  if (failed(loc))
    return failure();
  auto name = description.getString("op");
  auto info = name
                  ? RegisteredOperationName::lookup(*name, builder.getContext())
                  : std::nullopt;
  if (!info || *name == "func.func" || *name == "builtin.module")
    return error(*loc, "operation must name a registered non-function MLIR op");
  OperationState state(*loc, *name);
  if (auto args = description.getArray("args")) {
    for (const auto &argument : *args) {
      auto id = argument.getAsString();
      auto found = id ? values.find(*id) : values.end();
      if (found == values.end())
        return error(*loc,
                     "operand must reference a visible earlier SSA value");
      state.addOperands(found->second);
    }
  } else if (description.get("args")) {
    return error(*loc, "operation args must be an array");
  }
  SmallVector<StringRef> resultNames;
  if (auto results = description.getArray("results")) {
    for (const auto &result : *results) {
      auto object = result.getAsObject();
      auto id = object ? object->getString("name") : std::nullopt;
      auto typeName = object ? object->getString("type") : std::nullopt;
      auto parsedType = typeName ? type(*typeName) : Type{};
      if (!id || !isShaderSymbolName(*id) || !parsedType || values.count(*id) ||
          llvm::is_contained(resultNames, *id))
        return error(*loc,
                     "result requires a unique SSA name and supported type");
      resultNames.push_back(*id);
      state.addTypes(parsedType);
    }
  } else if (description.get("results")) {
    return error(*loc, "operation results must be an array");
  }
  if (auto successors = description.getArray("successors")) {
    if (!successors->empty() && !info->hasTrait<OpTrait::IsTerminator>())
      return error(*loc, "only terminators can have successor blocks");
    for (const auto &successor : *successors) {
      auto id = successor.getAsString();
      auto found = id ? blocks.find(*id) : blocks.end();
      if (found == blocks.end())
        return error(*loc,
                     "successor must reference a block in the same region");
      state.addSuccessors(found->second);
    }
  } else if (description.get("successors")) {
    return error(*loc, "successors must be an array");
  }
  auto regions = description.getArray("regions");
  if (!regions && description.get("regions"))
    return error(*loc, "regions must be an array");
  if (regions)
    for (size_t i = 0; i < regions->size(); ++i)
      state.addRegion();
  NamedAttrList attributes;
  if (auto attrs = description.getObject("attributes")) {
    for (const auto &item : *attrs) {
      if (!llvm::is_contained(info->getAttributeNames(),
                              builder.getStringAttr(StringRef(item.first))))
        return error(*loc, "unknown property for registered operation");
      auto attr = attribute(item.second, *loc);
      if (failed(attr))
        return failure();
      attributes.append(item.first, *attr);
    }
  } else if (description.get("attributes")) {
    return error(*loc, "attributes must be an object");
  }
  // Validate properties explicitly instead of using OperationState's asserting
  // propertiesAttr creation path for untrusted input.
  Operation *op = builder.create(state);
  if (info->getOpPropertyByteSize()) {
    if (failed(op->setPropertiesFromAttribute(
            attributes.getDictionary(builder.getContext()),
            [&] { return op->emitError("invalid operation properties: "); })))
      return failure();
  } else {
    op->setAttrs(attributes.getDictionary(builder.getContext()));
  }
  if (regions)
    for (auto [index, item] : llvm::enumerate(*regions)) {
      auto object = item.getAsObject();
      if (!object)
        return error(*loc, "region must be an object");
      OpBuilder::InsertionGuard guard(builder);
      if (failed(region(*object, op->getRegion(index), values, *loc)))
        return failure();
    }
  if (auto power = dyn_cast<math::PowFOp>(op)) {
    if (failed(verify(op)))
      return failure();
    Value value =
        buildPortablePow(builder, *loc, power.getLhs(), power.getRhs());
    values.try_emplace(resultNames.front(), value);
    power.erase();
  } else {
    for (auto [index, name] : llvm::enumerate(resultNames))
      values.try_emplace(name, op->getResult(index));
  }
  return success();
}

LogicalResult StructuredImporter::region(const llvm::json::Object &description,
                                         Region &region, const Values &captures,
                                         Location parent) {
  auto loc = location(description, parent);
  auto descriptions = description.getArray("blocks");
  if (failed(loc))
    return failure();
  if (!descriptions || descriptions->empty())
    return error(*loc, "region requires a nonempty blocks array");
  Blocks blocks;
  Values values(captures);
  for (const auto &item : *descriptions) {
    auto object = item.getAsObject();
    auto name = object ? object->getString("name") : std::nullopt;
    auto args = object ? object->getArray("arguments") : nullptr;
    if (!name || !isShaderSymbolName(*name) || blocks.count(*name) || !args)
      return error(*loc, "block requires a unique name and arguments array");
    auto blockLoc = location(*object, *loc);
    if (failed(blockLoc))
      return failure();
    auto block = new Block;
    region.push_back(block);
    blocks.try_emplace(*name, block);
    for (const auto &argument : *args) {
      auto arg = argument.getAsObject();
      auto id = arg ? arg->getString("name") : std::nullopt;
      auto name = arg ? arg->getString("type") : std::nullopt;
      auto parsedType = name ? type(*name) : Type{};
      if (!id || !isShaderSymbolName(*id) || !parsedType || values.count(*id))
        return error(*blockLoc, "block argument requires a unique SSA name "
                                "and supported type");
      values.try_emplace(*id, block->addArgument(parsedType, *blockLoc));
    }
  }
  for (const auto &item : *descriptions) {
    const auto &object = *item.getAsObject();
    auto blockLoc = location(object, *loc);
    auto operations = object.getArray("operations");
    if (!operations)
      return error(*blockLoc, "block requires an operations array");
    builder.setInsertionPointToEnd(blocks.lookup(*object.getString("name")));
    for (const auto &item : *operations) {
      auto op = item.getAsObject();
      if (!op)
        return error(*blockLoc, "operation must be an object");
      if (failed(operation(*op, values, blocks, *blockLoc)))
        return failure();
    }
  }
  return success();
}

OwningOpRef<ModuleOp>
StructuredImporter::import(const llvm::json::Object &source) {
  auto loc = location(source, builder.getUnknownLoc());
  if (failed(loc))
    return {};
  auto functions = source.getArray("functions");
  if (!functions || functions->empty()) {
    emitError(*loc, "structured module requires a nonempty functions array");
    return {};
  }
  auto module = OwningOpRef<ModuleOp>(ModuleOp::create(*loc));
  if (auto description = source.get("parameters")) {
    auto object = description->getAsObject();
    auto name = object ? object->getString("name") : std::nullopt;
    auto fields = object ? object->getArray("fields") : nullptr;
    if (!name || !fields || object->size() != 2) {
      emitError(*loc, "parameters requires a name and fields, not authored "
                      "offsets or backend bindings");
      return {};
    }
    std::function<FailureOr<Type>(const llvm::json::Object &)> parseParamType =
        [&](const llvm::json::Object &f) -> FailureOr<Type> {
      auto typeVal = f.get("type");
      if (!typeVal)
        return failure();
      if (auto typeStr = typeVal->getAsString()) {
        if (*typeStr == "array") {
          auto count = f.getInteger("count");
          if (!count || *count < 1 || *count > INT32_MAX)
            return failure();
          auto elemVal = f.get("element_type");
          if (!elemVal)
            return failure();
          Type elemType;
          if (auto elemStr = elemVal->getAsString()) {
            elemType = parseShaderType(*builder.getContext(), *elemStr);
          } else if (auto elemObj = elemVal->getAsObject()) {
            auto parsed = parseParamType(*elemObj);
            if (succeeded(parsed))
              elemType = *parsed;
          }
          if (!elemType)
            return failure();
          return ArrayType::get(builder.getContext(), elemType, static_cast<uint32_t>(*count));
        }
        if (*typeStr == "struct") {
          auto structFields = f.getArray("fields");
          if (!structFields)
            return failure();
          SmallVector<Attribute> nestedAttrs;
          for (const auto &subItem : *structFields) {
            auto subObj = subItem.getAsObject();
            if (!subObj)
              return failure();
            auto subName = subObj->getString("name");
            if (!subName || !isShaderSymbolName(*subName))
              return failure();
            auto subType = parseParamType(*subObj);
            if (failed(subType))
              return failure();
            nestedAttrs.push_back(ParameterFieldAttr::get(builder.getContext(), *subName, *subType));
          }
          auto structName = f.getString("name");
          return StructType::get(builder.getContext(), structName ? *structName : "struct",
                                 builder.getArrayAttr(nestedAttrs));
        }
        if (*typeStr == "u8" || *typeStr == "ui8")
          return IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned);
        if (*typeStr == "i8")
          return IntegerType::get(builder.getContext(), 8, IntegerType::Signed);
        if (*typeStr == "u16" || *typeStr == "ui16")
          return IntegerType::get(builder.getContext(), 16, IntegerType::Unsigned);
        if (*typeStr == "i16")
          return IntegerType::get(builder.getContext(), 16, IntegerType::Signed);
        if (*typeStr == "u32" || *typeStr == "ui32")
          return IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned);
        if (*typeStr == "i32")
          return IntegerType::get(builder.getContext(), 32, IntegerType::Signed);
        if (*typeStr == "u64" || *typeStr == "ui64")
          return IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
        if (*typeStr == "i64")
          return IntegerType::get(builder.getContext(), 64, IntegerType::Signed);
        if (typeStr->ends_with("x2") || typeStr->ends_with("x3") || typeStr->ends_with("x4")) {
          int64_t count = typeStr->back() - '0';
          StringRef prefix = typeStr->drop_back(2);
          llvm::json::Object subObj{{"type", prefix}};
          auto sub = parseParamType(subObj);
          if (succeeded(sub))
            return VectorType::get({count}, *sub);
        }
        auto parsed = parseShaderType(*builder.getContext(), *typeStr);
        if (!parsed)
          return failure();
        return parsed;
      }
      return failure();
    };

    SmallVector<Attribute> attributes;
    for (const auto &item : *fields) {
      auto field = item.getAsObject();
      auto name = field ? field->getString("name") : std::nullopt;
      if (!field || !name || field->get("offset") || field->get("binding")) {
        emitError(*loc,
                  "parameter field requires only a name and supported type");
        return {};
      }
      auto parsedType = parseParamType(*field);
      if (failed(parsedType)) {
        emitError(*loc,
                  "parameter field requires only a name and supported type");
        return {};
      }
      auto attribute = ParameterFieldAttr::getChecked(
          [&] { return emitError(*loc); }, builder.getContext(), *name,
          *parsedType);
      if (!attribute)
        return {};
      attributes.push_back(attribute);
    }
    auto block = ParameterBlockAttr::getChecked(
        [&] { return emitError(*loc); }, builder.getContext(), *name,
        builder.getArrayAttr(attributes));
    if (!block)
      return {};
    (*module)->setAttr("gfx.parameters", block);
  }
  llvm::StringMap<func::FuncOp> symbols;
  for (const auto &item : *functions) {
    auto object = item.getAsObject();
    auto name = object ? object->getString("name") : std::nullopt;
    auto results = object ? object->getArray("results") : nullptr;
    auto body = object ? object->getObject("body") : nullptr;
    auto blocks = body ? body->getArray("blocks") : nullptr;
    auto entry =
        blocks && !blocks->empty() ? (*blocks)[0].getAsObject() : nullptr;
    auto args = entry ? entry->getArray("arguments") : nullptr;
    if (!name || !isShaderSymbolName(*name) || symbols.count(*name) ||
        !results || !args) {
      emitError(*loc, "function requires a unique symbol, result types, and "
                      "body with typed entry-block arguments");
      return {};
    }
    auto functionLoc = location(*object, *loc);
    if (failed(functionLoc))
      return {};
    SmallVector<Type> inputs, outputs;
    for (const auto &item : *args) {
      auto arg = item.getAsObject();
      auto name = arg ? arg->getString("type") : std::nullopt;
      auto parsedType = name ? type(*name) : Type{};
      if (!parsedType) {
        emitError(*functionLoc, "function argument requires a supported type");
        return {};
      }
      inputs.push_back(parsedType);
    }
    for (const auto &item : *results) {
      auto name = item.getAsString();
      auto parsedType = name ? type(*name) : Type{};
      if (!parsedType) {
        emitError(*functionLoc, "function result requires a supported type");
        return {};
      }
      outputs.push_back(parsedType);
    }
    builder.setInsertionPointToEnd(module->getBody());
    auto function = func::FuncOp::create(
        builder, *functionLoc, *name, builder.getFunctionType(inputs, outputs));
    symbols.try_emplace(*name, function);
    if (auto stageName = object->getString("stage")) {
      auto stage = symbolizeStage(*stageName);
      if (!stage) {
        emitError(*functionLoc, "unknown entry stage");
        return {};
      }
      function->setAttr("gfx.stage",
                        StageAttr::get(builder.getContext(), *stage));
      if (auto size = object->get("workgroup_size")) {
        llvm::json::Object attr{{"kind", "i32_array"}, {"value", *size}};
        auto parsed =
            attribute(llvm::json::Value(std::move(attr)), *functionLoc);
        if (failed(parsed))
          return {};
        function->setAttr("gfx.workgroup_size", *parsed);
      }
    } else if (object->get("stage")) {
      emitError(*functionLoc, "stage must be a string");
      return {};
    } else {
      function.setPrivate();
      if (object->get("workgroup_size")) {
        emitError(*functionLoc, "workgroup_size requires a compute entry");
        return {};
      }
    }
  }
  for (const auto &item : *functions) {
    const auto &object = *item.getAsObject();
    auto function = symbols.lookup(*object.getString("name"));
    if (failed(region(*object.getObject("body"), function.getBody(), {},
                      function.getLoc())))
      return {};
  }
  if (failed(verifyShaderModule(*module)))
    return {};
  return module;
}
} // namespace

OwningOpRef<ModuleOp>
mlir::gfx::importStructuredShader(MLIRContext &context,
                                  const llvm::json::Object &source) {
  return StructuredImporter(context).import(source);
}
