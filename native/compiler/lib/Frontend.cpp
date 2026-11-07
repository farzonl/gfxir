#include "Gfx/Gfx.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::gfx;

static Type parseType(MLIRContext &context, StringRef name) {
  if (name == "f32")
    return Float32Type::get(&context);
  if (name == "u32")
    return IntegerType::get(&context, 32);
  if (name == "bool")
    return IntegerType::get(&context, 1);
  if (name == "f32x2" || name == "f32x3" || name == "f32x4")
    return VectorType::get({name.back() - '0'}, Float32Type::get(&context));
  return {};
}

static std::string typeName(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return typeName(vector.getElementType()) + "x" +
           std::to_string(vector.getNumElements());
  if (type.isF32())
    return "f32";
  return type.isInteger(1) ? "bool" : "u32";
}

static bool symbolName(StringRef name) {
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

OwningOpRef<ModuleOp>
mlir::gfx::importShader(MLIRContext &context,
                        const llvm::json::Object &source) {
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
  auto name = source.getString("name");
  auto stageName = source.getString("stage");
  auto stage = stageName ? symbolizeStage(*stageName) : std::nullopt;
  auto inputDescriptions = source.getArray("inputs");
  auto outputDescriptions = source.getArray("outputs");
  auto nodes = source.getArray("nodes");
  if (!name || !symbolName(*name) || !stage || !inputDescriptions ||
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
    Type type = typeString ? parseType(context, *typeString) : Type{};
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
          !symbolName(*label) || !mode)
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
    Type type = typeString ? parseType(context, *typeString) : Type{};
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
        if (integer && *integer >= 0 && *integer <= UINT32_MAX)
          literal = builder.getIntegerAttr(type, *integer);
      }
      if (!literal)
        return error("constant literal does not match its type");
      value = arith::ConstantOp::create(builder, location, type, literal);
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
      if (*operation == "eq") {
        if (type != builder.getI1Type() || isa<VectorType>(operandType))
          return error(
              "equality requires scalar operands and a boolean result");
        if (floating)
          value = arith::CmpFOp::create(builder, location,
                                        arith::CmpFPredicate::OEQ, arguments[0],
                                        arguments[1]);
        else
          value =
              arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq,
                                    arguments[0], arguments[1]);
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
        else
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
    if (!typeString || parseType(context, *typeString) != values[*id].getType())
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
          !symbolName(*label) || !mode)
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
  auto function = *module.getOps<func::FuncOp>().begin();
  ShaderInterface result{
      function.getSymName().str(),
      function->getAttrOfType<StageAttr>("gfx.stage").getValue(),
      {},
      {}};
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
  return llvm::json::Object{{"name", reflection.name},
                            {"stage", stringifyStage(reflection.stage)},
                            {"inputs", serialize(reflection.inputs)},
                            {"outputs", serialize(reflection.outputs)}};
}
