#include "Gfx/Gfx.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::gfx;

Type mlir::gfx::parseShaderType(MLIRContext &context, StringRef name) {
  if (name == "f16")
    return Float16Type::get(&context);
  if (name == "f32")
    return Float32Type::get(&context);
  if (name == "f64")
    return Float64Type::get(&context);
  if (name == "i8" || name == "u8" || name == "ui8")
    return IntegerType::get(&context, 8);
  if (name == "i16" || name == "u16" || name == "ui16")
    return IntegerType::get(&context, 16);
  if (name == "i32" || name == "u32" || name == "ui32")
    return IntegerType::get(&context, 32);
  if (name == "i64" || name == "u64" || name == "ui64")
    return IntegerType::get(&context, 64);
  if (name == "bool" || name == "i1")
    return IntegerType::get(&context, 1);

  if (name.ends_with("x2") || name.ends_with("x3") || name.ends_with("x4")) {
    int64_t count = name.back() - '0';
    StringRef prefix = name.drop_back(2);
    if (Type elemType = parseShaderType(context, prefix))
      return VectorType::get({count}, elemType);
  }
  return {};
}

static std::string typeName(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return typeName(vector.getElementType()) + "x" +
           std::to_string(vector.getNumElements());
  if (isa<ArrayType>(type))
    return "array";
  if (isa<StructType>(type))
    return "struct";
  if (type.isF16())
    return "f16";
  if (type.isF32())
    return "f32";
  if (type.isF64())
    return "f64";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 1)
      return "bool";
    if (intType.isUnsigned())
      return "u" + std::to_string(intType.getWidth());
    if (intType.isSigned())
      return "i" + std::to_string(intType.getWidth());
    return (intType.getWidth() == 32) ? "u32"
                                      : "i" + std::to_string(intType.getWidth());
  }
  return "unknown";
}

bool mlir::gfx::isShaderSymbolName(StringRef name) {
  if (name.empty() || !(llvm::isAlpha(name.front()) || name.front() == '_'))
    return false;
  return llvm::all_of(name.drop_front(),
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

static InterpolationAttr interpolation(OpBuilder &builder,
                                       const llvm::json::Object &io) {
  auto name = io.getString("interpolation");
  auto value = name ? symbolizeInterpolation(*name) : std::nullopt;
  return value ? InterpolationAttr::get(builder.getContext(), *value)
               : InterpolationAttr{};
}

Value mlir::gfx::buildPortablePow(OpBuilder &builder, Location location,
                                  Value base, Value exponent) {
  Type type = base.getType();
  auto constant = [&](float number) -> Value {
    TypedAttr literal = builder.getF32FloatAttr(number);
    if (auto vector = dyn_cast<VectorType>(type))
      literal = DenseElementsAttr::get(vector, ArrayRef<Attribute>{literal});
    return arith::ConstantOp::create(builder, location, type, literal);
  };
  auto zero = constant(0), one = constant(1);
  auto baseZero = arith::CmpFOp::create(builder, location,
                                        arith::CmpFPredicate::OEQ, base, zero);
  auto exponentPositive = arith::CmpFOp::create(
      builder, location, arith::CmpFPredicate::OGT, exponent, zero);
  auto zeroCase =
      arith::AndIOp::create(builder, location, baseZero, exponentPositive);
  // Upstream pow expansion uses log(base), whose portable domain excludes zero.
  auto safeBase =
      arith::SelectOp::create(builder, location, zeroCase, one, base);
  auto power = math::PowFOp::create(builder, location, safeBase, exponent);
  return arith::SelectOp::create(builder, location, zeroCase, zero, power);
}

OwningOpRef<ModuleOp>
mlir::gfx::importShader(MLIRContext &context,
                        const llvm::json::Object &source) {
  if (source.get("schema_version")) {
    auto version = source.getInteger("schema_version");
    if (version && *version == 2)
      return importStructuredShader(context, source);
    if (!version || *version != 1) {
      emitError(UnknownLoc::get(&context),
                "supported shader input versions are 1 (flat graph) and "
                "2 (structured module); migrate to a supported version");
      return {};
    }
  }
  OpBuilder builder(&context);
  Location location = builder.getUnknownLoc();
  if (auto file = source.getString("source_file")) {
    auto line = source.getInteger("source_line");
    if (!line || *line < 1 || *line > UINT32_MAX) {
      emitError(location, "source_line must be a positive 32-bit integer");
      return {};
    }
    location = FileLineColLoc::get(&context, *file, *line, 1);
  }
  auto error = [&](StringRef message) -> OwningOpRef<ModuleOp> {
    emitError(location, message);
    return {};
  };
  if (source.get("parameters"))
    return error("parameter blocks require structured input schema version 2");
  auto name = source.getString("name");
  auto stageName = source.getString("stage");
  auto stage = stageName ? symbolizeStage(*stageName) : std::nullopt;
  auto inputDescriptions = source.getArray("inputs");
  auto outputDescriptions = source.getArray("outputs");
  auto nodes = source.getArray("nodes");
  if (!name || !isShaderSymbolName(*name) || !stage || !inputDescriptions ||
      !outputDescriptions || !nodes)
    return error("shader requires a valid name, stage, inputs, outputs, and "
                 "typed node graph");
  auto module = OwningOpRef<ModuleOp>(ModuleOp::create(location));
  builder.setInsertionPointToStart(module->getBody());
  auto function = func::FuncOp::create(builder, location, *name,
                                       builder.getFunctionType({}, {}));
  function->setAttr("gfx.stage", StageAttr::get(&context, *stage));
  if (*stage == Stage::Compute) {
    auto size = source.getArray("workgroup_size");
    SmallVector<int32_t> dimensions;
    if (size)
      for (const auto &dimension : *size) {
        auto value = dimension.getAsInteger();
        if (!value || *value < 1 || *value > INT32_MAX)
          return error("workgroup dimensions must be positive 32-bit integers");
        dimensions.push_back(*value);
      }
    function->setAttr("gfx.workgroup_size",
                      builder.getDenseI32ArrayAttr(dimensions));
  }
  builder.setInsertionPointToStart(function.addEntryBlock());
  SmallVector<Value> inputs;
  for (const auto &description : *inputDescriptions) {
    auto io = description.getAsObject();
    auto typeString = io ? io->getString("type") : std::nullopt;
    Type type = typeString ? parseShaderType(context, *typeString) : Type{};
    if (!io || !type)
      return error("input requires a supported scalar/vector type");
    if (auto builtinName = io->getString("builtin")) {
      auto kind = symbolizeBuiltin(*builtinName);
      if (!kind)
        return error("unknown device builtin");
      inputs.push_back(BuiltinOp::create(builder, location, type,
                                         BuiltinAttr::get(&context, *kind)));
    } else {
      auto index = io->getInteger("location");
      auto label = io->getString("name");
      auto mode = interpolation(builder, *io);
      if (!index || *index < 0 || *index > INT32_MAX || !label ||
          !isShaderSymbolName(*label) || !mode)
        return error("input requires a nonnegative location, name, and "
                     "interpolation rule");
      inputs.push_back(InputOp::create(builder, location, type,
                                       builder.getI32IntegerAttr(*index),
                                       builder.getStringAttr(*label), mode));
    }
  }
  SmallVector<Value> values;
  for (const auto &description : *nodes) {
    auto node = description.getAsObject();
    auto operation = node ? node->getString("op") : std::nullopt;
    auto typeString = node ? node->getString("type") : std::nullopt;
    Type type = typeString ? parseShaderType(context, *typeString) : Type{};
    if (!node || !operation || !type)
      return error("node requires a known operation and scalar/vector type");
    SmallVector<Value> arguments;
    if (auto indices = node->getArray("args"))
      for (const auto &index : *indices) {
        auto id = index.getAsInteger();
        if (!id || *id < 0 || static_cast<uint64_t>(*id) >= values.size())
          return error("node operands must reference earlier nodes");
        arguments.push_back(values[*id]);
      }
    Value value;
    if (*operation == "input") {
      auto id = node->getInteger("input");
      if (!id || *id < 0 || static_cast<uint64_t>(*id) >= inputs.size() ||
          !arguments.empty())
        return error("input node must reference a declared input");
      value = inputs[*id];
    } else if (*operation == "constant") {
      if (!arguments.empty())
        return error("constant cannot have operands");
      TypedAttr literal;
      if (type.isF32()) {
        auto number = node->getNumber("literal");
        if (number && std::isfinite(*number) &&
            std::abs(*number) <= std::numeric_limits<float>::max())
          literal = builder.getF32FloatAttr(*number);
      } else if (type.isInteger(1)) {
        auto boolean = node->getBoolean("literal");
        if (boolean)
          literal = builder.getBoolAttr(*boolean);
      } else if (type.isInteger(32)) {
        auto integer = node->getInteger("literal");
        bool signedType = *typeString == "i32";
        if (integer && *integer >= (signedType ? INT32_MIN : 0) &&
            *integer <= (signedType ? int64_t(INT32_MAX) : int64_t(UINT32_MAX)))
          literal = builder.getIntegerAttr(type, *integer);
      }
      if (!literal)
        return error("constant literal does not match its type");
      value = arith::ConstantOp::create(builder, location, type, literal);
    } else if (*operation == "broadcast") {
      auto vector = dyn_cast<VectorType>(type);
      if (!vector || arguments.size() != 1 ||
          arguments[0].getType() != vector.getElementType())
        return error(
            "broadcast requires one matching scalar and a vector result");
      value =
          vector::BroadcastOp::create(builder, location, type, arguments[0]);
    } else if (*operation == "cast") {
      if (arguments.size() != 1 || !type.isInteger(32) ||
          !arguments[0].getType().isInteger(32))
        return error("integer reinterpretation requires i32/u32 scalars");
      value = arguments[0];
    } else if (*operation == "neg" || *operation == "abs") {
      if (arguments.size() != 1 || arguments[0].getType() != type)
        return error("unary operations require one matching operand");
      if (type.isF32() || isa<VectorType>(type))
        value =
            *operation == "neg"
                ? Value(arith::NegFOp::create(builder, location, arguments[0]))
                : Value(math::AbsFOp::create(builder, location, arguments[0]));
      else if (*typeString == "i32") {
        if (*operation == "abs")
          value = math::AbsIOp::create(builder, location, arguments[0]);
        else {
          auto zero = arith::ConstantOp::create(builder, location,
                                                builder.getI32IntegerAttr(0));
          value = arith::SubIOp::create(builder, location, zero, arguments[0]);
        }
      } else
        return error("neg/abs requires a floating-point or signed value");
    } else if (*operation == "sin" || *operation == "cos" ||
               *operation == "sqrt" || *operation == "exp" ||
               *operation == "exp2" || *operation == "log" ||
               *operation == "log2" || *operation == "floor") {
      if (arguments.size() != 1 || arguments[0].getType() != type ||
          !(type.isF32() || isa<VectorType>(type)))
        return error(
            "shader math requires one matching f32 scalar/vector operand");
      if (*operation == "sin")
        value = math::SinOp::create(builder, location, arguments[0]);
      else if (*operation == "cos")
        value = math::CosOp::create(builder, location, arguments[0]);
      else if (*operation == "sqrt")
        value = math::SqrtOp::create(builder, location, arguments[0]);
      else if (*operation == "exp")
        value = math::ExpOp::create(builder, location, arguments[0]);
      else if (*operation == "exp2")
        value = math::Exp2Op::create(builder, location, arguments[0]);
      else if (*operation == "log")
        value = math::LogOp::create(builder, location, arguments[0]);
      else if (*operation == "log2")
        value = math::Log2Op::create(builder, location, arguments[0]);
      else
        value = math::FloorOp::create(builder, location, arguments[0]);
    } else if (*operation == "vector") {
      auto vector = dyn_cast<VectorType>(type);
      if (!vector ||
          static_cast<int64_t>(arguments.size()) != vector.getNumElements() ||
          llvm::any_of(arguments, [&](Value argument) {
            return argument.getType() != vector.getElementType();
          }))
        return error(
            "vector components must match the declared shape and element type");
      value =
          vector::FromElementsOp::create(builder, location, type, arguments);
    } else if (*operation == "extract") {
      auto index = node->getInteger("literal");
      auto vector = arguments.size() == 1
                        ? dyn_cast<VectorType>(arguments[0].getType())
                        : VectorType{};
      if (!vector || !index || *index < 0 || *index >= vector.getNumElements())
        return error("extract requires an in-range vector component");
      value = vector::ExtractOp::create(builder, location, arguments[0],
                                        ArrayRef<int64_t>{*index});
    } else if (*operation == "select") {
      if (arguments.size() != 3 || !arguments[0].getType().isInteger(1) ||
          arguments[1].getType() != type || arguments[2].getType() != type)
        return error(
            "select requires a boolean condition and matching result types");
      value = arith::SelectOp::create(builder, location, arguments[0],
                                      arguments[1], arguments[2]);
    } else {
      if (arguments.size() != 2 ||
          arguments[0].getType() != arguments[1].getType())
        return error("binary operations require two matching operands");
      auto operandType = arguments[0].getType();
      bool floating = operandType.isF32() || isa<VectorType>(operandType);
      if (*operation == "eq" || *operation == "cmp") {
        if (type != builder.getI1Type() || isa<VectorType>(operandType))
          return error(
              "equality requires scalar operands and a boolean result");
        if (*operation == "cmp") {
          auto predicate = node->getString("literal");
          if (!predicate)
            return error("comparison requires a typed predicate");
          if (floating) {
            auto kind = arith::symbolizeCmpFPredicate(*predicate);
            if (!kind)
              return error("invalid floating-point comparison predicate");
            value = arith::CmpFOp::create(builder, location, *kind,
                                          arguments[0], arguments[1]);
          } else {
            auto kind = arith::symbolizeCmpIPredicate(*predicate);
            if (!kind || (operandType.isInteger(1) &&
                          *kind != arith::CmpIPredicate::eq &&
                          *kind != arith::CmpIPredicate::ne))
              return error("invalid integer/boolean comparison predicate");
            value = arith::CmpIOp::create(builder, location, *kind,
                                          arguments[0], arguments[1]);
          }
        } else if (floating)
          value = arith::CmpFOp::create(builder, location,
                                        arith::CmpFPredicate::OEQ, arguments[0],
                                        arguments[1]);
        else
          value =
              arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                    arguments[0], arguments[1]);
      } else if (*operation == "and" || *operation == "or" ||
                 *operation == "xor") {
        if (!type.isInteger(1) || operandType != type)
          return error(
              "boolean operations require boolean operands and result");
        if (*operation == "and")
          value = arith::AndIOp::create(builder, location, arguments[0],
                                        arguments[1]);
        else if (*operation == "or")
          value = arith::OrIOp::create(builder, location, arguments[0],
                                       arguments[1]);
        else
          value = arith::XOrIOp::create(builder, location, arguments[0],
                                        arguments[1]);
      } else {
        if (operandType != type || type.isInteger(1))
          return error(
              "arithmetic requires matching numeric operands and result type");
        if (*operation == "add")
          value = floating
                      ? Value(arith::AddFOp::create(builder, location,
                                                    arguments[0], arguments[1]))
                      : Value(arith::AddIOp::create(
                            builder, location, arguments[0], arguments[1]));
        else if (*operation == "sub")
          value = floating
                      ? Value(arith::SubFOp::create(builder, location,
                                                    arguments[0], arguments[1]))
                      : Value(arith::SubIOp::create(
                            builder, location, arguments[0], arguments[1]));
        else if (*operation == "mul")
          value = floating
                      ? Value(arith::MulFOp::create(builder, location,
                                                    arguments[0], arguments[1]))
                      : Value(arith::MulIOp::create(
                            builder, location, arguments[0], arguments[1]));
        else if (*operation == "div") {
          if (!floating)
            return error("division currently requires floating-point operands");
          value = arith::DivFOp::create(builder, location, arguments[0],
                                        arguments[1]);
        } else if (*operation == "pow") {
          if (!floating)
            return error("pow requires matching f32 scalar/vector operands");
          value =
              buildPortablePow(builder, location, arguments[0], arguments[1]);
        } else if (*operation == "min" || *operation == "max") {
          if (floating)
            value = *operation == "min"
                        ? Value(arith::MinNumFOp::create(
                              builder, location, arguments[0], arguments[1]))
                        : Value(arith::MaxNumFOp::create(
                              builder, location, arguments[0], arguments[1]));
          else if (*typeString == "i32")
            value = *operation == "min"
                        ? Value(arith::MinSIOp::create(
                              builder, location, arguments[0], arguments[1]))
                        : Value(arith::MaxSIOp::create(
                              builder, location, arguments[0], arguments[1]));
          else
            value = *operation == "min"
                        ? Value(arith::MinUIOp::create(
                              builder, location, arguments[0], arguments[1]))
                        : Value(arith::MaxUIOp::create(
                              builder, location, arguments[0], arguments[1]));
        } else
          return error("unsupported node operation");
      }
    }
    if (value.getType() != type)
      return error("node result does not match its declared type");
    values.push_back(value);
  }
  for (const auto &description : *outputDescriptions) {
    auto io = description.getAsObject();
    auto id = io ? io->getInteger("value") : std::nullopt;
    if (!io || !id || *id < 0 || static_cast<uint64_t>(*id) >= values.size())
      return error("output must reference a valid graph value");
    auto typeString = io->getString("type");
    if (!typeString ||
        parseShaderType(context, *typeString) != values[*id].getType())
      return error("output type does not match its value");
    if (auto builtin = io->getString("builtin")) {
      if (*builtin != "position")
        return error("unsupported output builtin");
      PositionOp::create(builder, location, values[*id]);
    } else {
      auto index = io->getInteger("location");
      auto label = io->getString("name");
      auto mode = interpolation(builder, *io);
      if (!index || *index < 0 || *index > INT32_MAX || !label ||
          !isShaderSymbolName(*label) || !mode)
        return error("output requires a nonnegative location, name, and "
                     "interpolation rule");
      OutputOp::create(builder, location, values[*id],
                       builder.getI32IntegerAttr(*index),
                       builder.getStringAttr(*label), mode);
    }
  }
  func::ReturnOp::create(builder, location);
  if (failed(verifyShaderModule(*module)))
    return {};
  return module;
}

ShaderInterface mlir::gfx::reflectShader(ModuleOp module) {
  func::FuncOp function;
  for (auto candidate : module.getOps<func::FuncOp>())
    if (candidate->hasAttr("gfx.stage"))
      function = candidate;
  ShaderInterface result{
      function.getSymName().str(),
      function->getAttrOfType<StageAttr>("gfx.stage").getValue(),
      {},
      {},
      {}};
  SmallVector<func::FuncOp> pending{function};
  llvm::SmallPtrSet<Operation *, 8> visited;
  bool reachedParamOp = false;
  while (!pending.empty()) {
    auto current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    current.walk([&](ParameterOp) {
      reachedParamOp = true;
      result.parameters =
          module->getAttrOfType<ParameterBlockAttr>("gfx.parameters");
    });
    current.walk([&](func::CallOp call) {
      if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        pending.push_back(callee);
    });
  }
  if (!reachedParamOp) {
    bool anyParamOpInModule = false;
    module.walk([&](ParameterOp) { anyParamOpInModule = true; });
    if (!anyParamOpInModule) {
      result.parameters =
          module->getAttrOfType<ParameterBlockAttr>("gfx.parameters");
    }
  }
  function.walk([&](Operation *op) {
    if (auto builtin = dyn_cast<BuiltinOp>(op)) {
      result.inputs.push_back({stringifyBuiltin(builtin.getKind()).str(),
                               builtin.getValue().getType(), builtin.getKind(),
                               Interpolation::Perspective});
    } else if (auto input = dyn_cast<InputOp>(op)) {
      result.inputs.push_back({input.getName().str(),
                               input.getValue().getType(),
                               static_cast<int32_t>(input.getLocation()),
                               input.getInterpolation()});
    } else if (auto output = dyn_cast<OutputOp>(op)) {
      result.outputs.push_back({output.getName().str(),
                                output.getValue().getType(),
                                static_cast<int32_t>(output.getLocation()),
                                output.getInterpolation()});
    } else if (auto position = dyn_cast<PositionOp>(op)) {
      result.outputs.push_back({"position", position.getValue().getType(),
                                PositionBinding{}, Interpolation::Perspective});
    }
  });
  return result;
}

llvm::json::Object
mlir::gfx::serializeReflection(const ShaderInterface &reflection) {
  auto serialize = [](ArrayRef<InterfaceElement> elements) {
    llvm::json::Array result;
    for (const auto &element : elements) {
      llvm::json::Object io{{"name", element.name},
                            {"type", typeName(element.type)}};
      if (auto builtin = std::get_if<Builtin>(&element.binding))
        io["builtin"] = stringifyBuiltin(*builtin);
      else if (auto location = std::get_if<int32_t>(&element.binding)) {
        io["location"] = *location;
        io["interpolation"] = stringifyInterpolation(element.interpolation);
      } else
        io["builtin"] = "position";
      result.push_back(std::move(io));
    }
    return result;
  };
  llvm::json::Object result{{"name", reflection.name},
                            {"stage", stringifyStage(reflection.stage)},
                            {"inputs", serialize(reflection.inputs)},
                            {"outputs", serialize(reflection.outputs)}};
  if (reflection.parameters)
    result["parameters"] =
        serializeParameters(reflection.parameters, reflection.stage);
  return result;
}

static llvm::json::Object serializeField(const ParameterFieldLayout &item,
                                         bool isLegacyF32) {
  llvm::json::Object fieldObj{
      {"name", item.name},
      {"type", typeName(item.type)},
      {"offset", static_cast<int64_t>(item.offset)},
      {"size", static_cast<int64_t>(item.size)},
      {"alignment", static_cast<int64_t>(item.alignment)}};
  if (!isLegacyF32) {
    if (auto arr = dyn_cast<ArrayType>(item.type)) {
      fieldObj["count"] = static_cast<int64_t>(arr.getCount());
      fieldObj["stride"] = static_cast<int64_t>(item.stride);
      fieldObj["element_type"] = typeName(arr.getElementType());
      if (auto elemSt = dyn_cast<StructType>(arr.getElementType())) {
        llvm::json::Array subFields;
        for (const auto &sub : item.nestedFields)
          subFields.push_back(serializeField(sub, false));
        fieldObj["fields"] = std::move(subFields);
      }
    } else if (auto st = dyn_cast<StructType>(item.type)) {
      llvm::json::Array subFields;
      for (const auto &sub : item.nestedFields)
        subFields.push_back(serializeField(sub, false));
      fieldObj["fields"] = std::move(subFields);
    }
  }
  return fieldObj;
}

llvm::json::Object mlir::gfx::serializeParameters(ParameterBlockAttr block,
                                                  Stage stage) {
  auto layout = parameterLayout(block);
  bool isLegacyF32 = (stage != Stage::Compute);
  for (const auto &item : layout.fields) {
    if (auto vec = dyn_cast<VectorType>(item.type)) {
      if (!vec.getElementType().isF32()) {
        isLegacyF32 = false;
        break;
      }
    } else if (!item.type.isF32()) {
      isLegacyF32 = false;
      break;
    }
  }
  if (layout.size > 128)
    isLegacyF32 = false;

  llvm::json::Array fields;
  for (const auto &item : layout.fields) {
    fields.push_back(serializeField(item, isLegacyF32));
  }

  if (isLegacyF32) {
    uint32_t legacySize = (layout.size + 15) / 16 * 16;
    return llvm::json::Object{
        {"schema_version", 1},
        {"name", block.getName()},
        {"layout", "gfx-root-f32-v1"},
        {"size", static_cast<int64_t>(legacySize)},
        {"alignment", 16},
        {"fields", std::move(fields)},
        {"stages", llvm::json::Array{stringifyStage(stage)}},
        {"ownership", "host"},
        {"access", "read-only"},
        {"addressable", false},
        {"bindings",
         llvm::json::Object{
             {"vulkan",
              llvm::json::Object{{"storage", "push_constant"}, {"offset", 0}}},
             {"directx",
              llvm::json::Object{{"storage", "root_constants"},
                                 {"parameter_index", 0},
                                 {"register", 0},
                                 {"space", 0},
                                 {"num32bit_constants", static_cast<int64_t>(legacySize / 4)},
                                 {"visibility", "all"},
                                 {"root_signature_version", "1.1"}}},
             {"metal",
              llvm::json::Object{{"storage", "converter_argument_buffer"},
                                 {"buffer_index", 2},
                                 {"offset", 0},
                                 {"size", static_cast<int64_t>(legacySize)}}}}}};
  }

  llvm::json::Object bindings;
  if (stage == Stage::Compute) {
    bindings = llvm::json::Object{
        {"vulkan",
         llvm::json::Object{{"storage", "storage_buffer"}, {"set", 0}, {"binding", 0}}},
        {"directx",
         llvm::json::Object{{"storage", "structured_buffer"}, {"register", 0}, {"space", 0}}},
        {"metal",
         llvm::json::Object{{"storage", "buffer"}, {"buffer_index", 0}}}};
  } else {
    bindings = llvm::json::Object{
        {"vulkan",
         llvm::json::Object{{"storage", "push_constant"}, {"offset", 0}}},
        {"directx",
         llvm::json::Object{{"storage", "root_constants"},
                            {"parameter_index", 0},
                            {"register", 0},
                            {"space", 0},
                            {"num32bit_constants", static_cast<int64_t>((layout.size + 3) / 4)},
                            {"visibility", "all"},
                            {"root_signature_version", "1.1"}}},
        {"metal",
         llvm::json::Object{{"storage", "converter_argument_buffer"},
                            {"buffer_index", 2},
                            {"offset", 0},
                            {"size", static_cast<int64_t>(layout.size)}}}};
  }

  return llvm::json::Object{
      {"schema_version", 2},
      {"name", block.getName()},
      {"layout", "gfx-std430-v1"},
      {"size", static_cast<int64_t>(layout.size)},
      {"alignment", static_cast<int64_t>(layout.alignment)},
      {"fields", std::move(fields)},
      {"stages", llvm::json::Array{stringifyStage(stage)}},
      {"ownership", "host"},
      {"access", "read-only"},
      {"addressable", false},
      {"bindings", std::move(bindings)}};
}

static std::string cTypeNameForField(Type type, StringRef name, uint32_t &arrCount, bool &isVec3) {
  arrCount = 0;
  isVec3 = false;
  if (type.isF16())
    return "uint16_t";
  if (type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return "uint8_t";
    if (intType.isUnsigned())
      return "uint" + std::to_string(width) + "_t";
    return "int" + std::to_string(width) + "_t";
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    int64_t count = vecType.getNumElements();
    if (count == 3)
      isVec3 = true;
    arrCount = static_cast<uint32_t>(count);
    uint32_t subArr = 0;
    bool subVec3 = false;
    return cTypeNameForField(vecType.getElementType(), name, subArr, subVec3);
  }
  if (auto st = dyn_cast<StructType>(type))
    return st.getName().str();
  return "uint8_t";
}

static void emitStructBody(ArrayRef<ParameterFieldLayout> fields,
                           uint32_t structSize,
                           llvm::raw_ostream &os,
                           StringRef structName) {
  uint32_t currentCursor = 0;
  uint32_t padIndex = 0;
  for (const auto &item : fields) {
    if (item.offset > currentCursor) {
      uint32_t gap = item.offset - currentCursor;
      os << "    uint8_t _pad" << padIndex++ << "[" << gap << "];\n";
      currentCursor = item.offset;
    }
    if (auto arr = dyn_cast<ArrayType>(item.type)) {
      uint32_t elemArr = 0;
      bool elemVec3 = false;
      std::string elemBase = cTypeNameForField(arr.getElementType(), item.name, elemArr, elemVec3);
      if (elemVec3) {
        os << "    struct { " << elemBase << " v[3]; uint8_t _p[4]; } "
           << item.name << "[" << arr.getCount() << "];\n";
      } else if (elemArr > 0) {
        os << "    " << elemBase << " " << item.name << "[" << arr.getCount()
           << "][" << elemArr << "];\n";
      } else {
        os << "    " << elemBase << " " << item.name << "[" << arr.getCount()
           << "];\n";
      }
    } else if (auto vec = dyn_cast<VectorType>(item.type)) {
      uint32_t vecCount = 0;
      bool isVec3 = false;
      std::string baseType = cTypeNameForField(vec, item.name, vecCount, isVec3);
      os << "    " << baseType << " " << item.name << "[" << vecCount << "];\n";
    } else if (auto st = dyn_cast<StructType>(item.type)) {
      os << "    " << st.getName() << " " << item.name << ";\n";
    } else {
      uint32_t dummy = 0;
      bool dummyVec3 = false;
      std::string scalarType = cTypeNameForField(item.type, item.name, dummy, dummyVec3);
      os << "    " << scalarType << " " << item.name << ";\n";
    }
    currentCursor = item.offset + item.size;
  }
  if (currentCursor < structSize) {
    uint32_t gap = structSize - currentCursor;
    os << "    uint8_t _pad" << padIndex++ << "[" << gap << "];\n";
  }
}

static void collectStructTypes(Type type, SmallVectorImpl<StructType> &structs,
                               llvm::DenseSet<Type> &visited) {
  if (auto arr = dyn_cast<ArrayType>(type)) {
    collectStructTypes(arr.getElementType(), structs, visited);
    return;
  }
  if (auto st = dyn_cast<StructType>(type)) {
    if (visited.contains(st))
      return;
    for (auto attr : st.getFields()) {
      auto field = cast<ParameterFieldAttr>(attr);
      collectStructTypes(field.getType(), structs, visited);
    }
    if (visited.insert(st).second) {
      structs.push_back(st);
    }
  }
}

LogicalResult mlir::gfx::emitCHeader(const ParameterLayout &layout,
                                     llvm::raw_ostream &os,
                                     llvm::StringRef structName) {
  std::string upperName = structName.upper();
  os << "/* Auto-generated by gfxir compiler std430 reflection engine. Do not edit. */\n";
  os << "#ifndef " << upperName << "_H\n";
  os << "#define " << upperName << "_H\n\n";
  os << "#pragma once\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <assert.h>\n\n";
  os << "#if !defined(__cplusplus) && !defined(static_assert)\n";
  os << "#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L\n";
  os << "#    define static_assert _Static_assert\n";
  os << "#  endif\n";
  os << "#endif\n\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n\n";

  // Collect and emit all referenced structs in topological order (innermost first)
  SmallVector<StructType, 8> referencedStructs;
  llvm::DenseSet<Type> visitedStructs;
  for (const auto &item : layout.fields) {
    collectStructTypes(item.type, referencedStructs, visitedStructs);
  }

  llvm::StringSet<> emittedNames;
  for (auto st : referencedStructs) {
    if (st.getName() == structName || !emittedNames.insert(st.getName()).second)
      continue;
    auto members = layoutStructMembers(st);
    uint32_t stSize = 0, stAlign = 0;
    (void)getStd430Layout(st, stSize, stAlign);

    os << "typedef struct " << st.getName() << " {\n";
    emitStructBody(members, stSize, os, st.getName());
    os << "} " << st.getName() << ";\n\n";

    for (const auto &sub : members) {
      os << "static_assert(offsetof(" << st.getName() << ", " << sub.name
         << ") == " << sub.offset << ", \"offset mismatch for " << sub.name << "\");\n";
    }
    os << "static_assert(sizeof(" << st.getName() << ") == " << stSize
       << ", \"size mismatch for " << st.getName() << "\");\n\n";
  }

  // Primary struct
  os << "typedef struct " << structName << " {\n";
  emitStructBody(layout.fields, layout.size, os, structName);
  os << "} " << structName << ";\n\n";

  // Compile-time static assertions
  for (const auto &item : layout.fields) {
    os << "static_assert(offsetof(" << structName << ", " << item.name
       << ") == " << item.offset << ", \"offset mismatch for " << item.name << "\");\n";
  }
  os << "static_assert(sizeof(" << structName << ") == " << layout.size
     << ", \"size mismatch for " << structName << "\");\n\n";

  os << "#ifdef __cplusplus\n";
  os << "}\n";
  os << "#endif\n\n";
  os << "#endif // " << upperName << "_H\n";
  return success();
}

LogicalResult mlir::gfx::emitCHeader(const ShaderInterface &reflection,
                                     llvm::raw_ostream &os) {
  if (reflection.parameters) {
    auto layout = parameterLayout(reflection.parameters);
    return emitCHeader(layout, os, reflection.parameters.getName());
  }
  os << "/* Auto-generated by gfx-compile std430 reflection engine. Do not edit. */\n";
  os << "#ifndef GFX_NO_PARAMETERS_H\n";
  os << "#define GFX_NO_PARAMETERS_H\n\n";
  os << "#pragma once\n\n";
  os << "/* No aggregate parameter blocks reflected. */\n\n";
  os << "#endif // GFX_NO_PARAMETERS_H\n";
  return success();
}
