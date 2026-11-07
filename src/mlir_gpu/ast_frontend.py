from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Callable

from . import sdf, shader_math
from .frontend import Entry, Node, Pipeline, Type, Value, Vertex, _node, maximum, minimum, select
from .frontend import i32, u32, vec2, vec3, vec4


_INTRINSICS = {Node, Value, Vertex, i32, u32, vec2, vec3, vec4, select, maximum, minimum}
for _module in (shader_math, sdf):
    _INTRINSICS.update(
        value for name, value in vars(_module).items()
        if not name.startswith("_") and isinstance(value, FunctionType)
        and value.__module__ == _module.__name__
    )


@dataclass(frozen=True)
class _Ref:
    name: str
    type: str


@dataclass
class _LoopExit:
    values: dict[str, Value]
    mode: int | Value
    returned: object = None


@dataclass(frozen=True)
class _Next:
    action: Callable[[], object]


class _Builder:
    def __init__(self, engine, arguments=(), captures=None):
        self.engine = engine
        self.block = {"name": "entry", "arguments": list(arguments), "operations": []}
        self.cache = dict(captures or {})

    def raw(self, op, args=(), types=(), attributes=None, regions=()):
        refs = [_Ref(self.engine.name(), typ) for typ in types]
        item = {
            "op": op, "args": [arg.name for arg in args],
            "results": [{"name": ref.name, "type": ref.type} for ref in refs],
            **self.engine.location(),
        }
        if attributes:
            item["attributes"] = attributes
        if regions:
            item["regions"] = [{"blocks": [region.block]} for region in regions]
        self.block["operations"].append(item)
        return refs

    def bind(self, ref, typ):
        node = Node("ssa", typ, literal=ref.name)
        self.cache[id(node)] = ref
        self.engine.owned.add(id(node))
        self.engine.nodes[id(node)] = node
        return Value(node)

    def value(self, op, args, typ, attributes=None):
        ref, = self.raw(op, [self.emit(arg.node) for arg in args], [typ.schema_name], attributes)
        return self.bind(ref, typ)

    def emit(self, node):
        key = id(node)
        if key in self.cache:
            return self.cache[key]
        self.engine.nodes[key] = node
        if node.op in ("input", "ssa"):
            raise ValueError("A stage cannot capture another stage's device input or a region-local SSA value.")
        args = [self.emit(arg) for arg in node.args]
        typ = node.type.schema_name
        attrs = None
        floating = node.args and node.args[0].type in (Type.F32, Type.VEC2, Type.VEC3, Type.VEC4)
        signed = node.args and node.args[0].type == Type.I32
        if node.op == "constant":
            op, attrs = "arith.constant", {"value": {"kind": "literal", "type": typ, "value": node.literal}}
        elif node.op in ("vec2", "vec3", "vec4"):
            op = "vector.from_elements"
        elif node.op == "extract":
            op, attrs = "vector.extract", {"static_position": {"kind": "i64_array", "value": [node.literal]}}
        elif node.op in ("eq", "cmp"):
            op = "arith.cmpf" if floating else "arith.cmpi"
            attrs = {"predicate": {"kind": "cmpf" if floating else "cmpi",
                                    "value": node.literal if node.op == "cmp" else "oeq" if floating else "eq"}}
        elif node.op in ("add", "sub", "mul"):
            op = f"arith.{node.op}{'f' if floating else 'i'}"
        elif node.op in ("min", "max"):
            op = f"arith.{node.op}{'numf' if floating else 'si' if signed else 'ui'}"
        elif node.op == "cast":
            self.cache[key] = args[0]
            return args[0]
        else:
            op = {
                "broadcast": "vector.broadcast", "select": "arith.select",
                "div": "arith.divf", "neg": "arith.negf" if floating else "arith.subi",
                "abs": "math.absf" if floating else "math.absi",
                "and": "arith.andi", "or": "arith.ori", "xor": "arith.xori",
                **{name: f"math.{name}" for name in ("sin", "cos", "sqrt", "exp", "exp2", "log", "log2", "floor")},
                "pow": "math.powf",
            }.get(node.op)
            if op is None:
                raise ValueError(f"Unsupported device operation: {node.op}.")
            if node.op == "neg" and not floating:
                args.insert(0, self.emit(_node(0, node.type)))
        ref, = self.raw(op, args, [typ], attrs)
        self.cache[key] = ref
        return ref


class ASTLowering:
    def __init__(self, function, inputs, interfaces):
        self.serial = 0
        self.owned = set()
        self.nodes = {id(node): node for node in inputs}
        self.structured = False
        self.stack = []
        self.scope = object()
        self.file = function.__code__.co_filename
        self.line = function.__code__.co_firstlineno
        self.column = 1
        self.builder = _Builder(self)
        self.root = self.builder
        for node, interface in zip(inputs, interfaces):
            attrs = (
                {"kind": {"kind": "builtin", "value": interface.builtin}}
                if interface.builtin else
                {"location": {"kind": "literal", "type": "i32", "value": interface.location},
                 "name": {"kind": "string", "value": interface.name},
                 "interpolation": {"kind": "interpolation", "value": interface.interpolation}}
            )
            ref, = self.root.raw("gfx.builtin" if interface.builtin else "gfx.input",
                                 types=[node.type.schema_name], attributes=attrs)
            self.root.cache[id(node)] = ref
        self.result = self.call(function, [Value(node) for node in inputs], {})

    def name(self):
        result = f"v{self.serial}"
        self.serial += 1
        return result

    def location(self):
        return {"source_file": self.file, "source_line": self.line, "source_column": self.column}

    def fail(self, message):
        error = TypeError(f"{self.file}:{self.line}:{self.column}: {message}")
        error._gfx_source_location = True
        raise error

    def located(self, action):
        try:
            return action()
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as error:
            if getattr(error, "_gfx_source_location", False):
                raise
            located = type(error)(f"{self.file}:{self.line}:{self.column}: {error}")
            located._gfx_source_location = True
            raise located from error

    def resolve(self, value):
        while isinstance(value, _Next):
            value = value.action()
        return value

    def lift(self, value, typ=None):
        if isinstance(value, Value):
            return Value(_node(value, typ))
        if typ is None:
            typ = Type.BOOL if isinstance(value, bool) else Type.I32 if isinstance(value, int) else Type.F32
        return Value(_node(value, typ))

    def pair(self, left, right):
        if isinstance(left, _LoopExit) and isinstance(right, _LoopExit):
            a, b = left.returned, right.returned
            if a is None and b is not None:
                a = self.zero(b)
            if b is None and a is not None:
                b = self.zero(a)
            values, rebuild_values = self.pair(left.values, right.values)
            modes, rebuild_modes = self.pair(self.lift(left.mode, Type.I32), self.lift(right.mode, Type.I32))
            returns, rebuild_returns = self.pair(a, b)
            count = len(values)
            return values + modes + returns, lambda items: _LoopExit(
                rebuild_values(items[:count]), rebuild_modes(items[count:count + 1]),
                rebuild_returns(items[count + 1:]),
            )
        if isinstance(left, Vertex) and isinstance(right, Vertex):
            values, rebuild = self.pair(
                (left.position, left.color, dict(left.varyings)),
                (right.position, right.color, dict(right.varyings)),
            )
            return values, lambda items: Vertex(*rebuild(items))
        if isinstance(left, dict) and isinstance(right, dict):
            if list(left) != list(right):
                self.fail("Device branches must define the same value/field names.")
            values, rebuild = self.pair(tuple(left.values()), tuple(right.values()))
            return values, lambda items: dict(zip(left, rebuild(items)))
        if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
            pairs = [self.pair(a, b) for a, b in zip(left, right)]
            values = [item for items, _ in pairs for item in items]
            def rebuild(items):
                result, offset = [], 0
                for leaves, callback in pairs:
                    result.append(callback(items[offset:offset + len(leaves)]))
                    offset += len(leaves)
                return tuple(result)
            return values, rebuild
        if left is None and right is None:
            return [], lambda items: None
        typ = left.node.type if isinstance(left, Value) else right.node.type if isinstance(right, Value) else None
        try:
            a, b = self.lift(left, typ), self.lift(right, typ)
        except (TypeError, ValueError) as error:
            self.fail(f"Incompatible device branch values: {error}")
        if a.node.type != b.node.type:
            self.fail("Device branches must merge matching scalar/vector types.")
        return [(a, b)], lambda items: items[0]

    def zero(self, value):
        if isinstance(value, Value):
            if value.node.type in (Type.VEC2, Type.VEC3, Type.VEC4):
                from .frontend import vec2, vec3, vec4
                return {Type.VEC2: vec2, Type.VEC3: vec3, Type.VEC4: vec4}[value.node.type](0)
            return self.lift(False if value.node.type == Type.BOOL else 0, value.node.type)
        if isinstance(value, Vertex):
            return Vertex(self.zero(value.position), self.zero(value.color), self.zero(dict(value.varyings)))
        if isinstance(value, dict):
            return {key: self.zero(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.zero(item) for item in value)
        return None if value is None else self.zero(self.lift(value))

    def choose(self, condition, yes, no):
        if not isinstance(condition, Value):
            if type(condition) not in (bool, int, float, str, tuple, dict, type(None)):
                self.fail("Compile-time conditions require scalar or literal collection constants.")
            return yes() if condition else no()
        if condition.node.type != Type.BOOL:
            self.fail("Device conditions require bool values, not numeric truthiness.")
        self.structured = True
        parent = self.builder
        cond = parent.emit(condition.node)
        branches = [_Builder(self, captures=parent.cache) for _ in range(2)]
        try:
            self.builder = branches[0]
            left = self.resolve(yes())
            self.builder = branches[1]
            right = self.resolve(no())
            leaves, rebuild = self.pair(left, right)
            types = []
            for index, branch in enumerate(branches):
                args = [branch.emit(pair[index].node) for pair in leaves]
                branch.raw("scf.yield", args)
                if index == 0:
                    types = [value.node.type for value, _ in leaves]
        finally:
            self.builder = parent
        refs = parent.raw("scf.if", [cond], [typ.schema_name for typ in types], regions=branches)
        return rebuild([parent.bind(ref, typ) for ref, typ in zip(refs, types)])

    def call(self, function, args, kwargs):
        if not callable(function) and not isinstance(function, (_Lambda, _Function)):
            self.fail("Calls must resolve to static Python helpers or registered shader intrinsics.")
        if function in (builtins.abs, minimum, maximum, builtins.min, builtins.max):
            if function is builtins.abs and (len(args) != 1 or type(args[0]) not in (Value, int, float, bool)):
                self.fail("abs requires a scalar or device numeric operand.")
            if function in (builtins.min, builtins.max) and any(isinstance(arg, Value) for arg in args):
                if len(args) != 2 or kwargs:
                    self.fail("Device min/max require two scalar/vector operands.")
                function = minimum if function is builtins.min else maximum
            return function(*args, **kwargs)
        if function in (builtins.int, builtins.float) and len(args) == 1 and isinstance(args[0], Value):
            value = args[0]
            if kwargs:
                self.fail("Device numeric conversions take one positional operand.")
            target = Type.I32 if function is builtins.int else Type.F32
            if value.node.type == target:
                return value
            op = {
                (Type.F32, Type.I32): "arith.fptosi",
                (Type.I32, Type.F32): "arith.sitofp",
                (Type.U32, Type.F32): "arith.uitofp",
            }.get((value.node.type, target))
            if op is None:
                self.fail("Unsupported explicit scalar numeric conversion.")
            self.structured = True
            return self.builder.value(op, [value], target)
        module = getattr(function, "__module__", "")
        bound = getattr(function, "__self__", None)
        intrinsic = function in _INTRINSICS if isinstance(function, (FunctionType, type)) else False
        if intrinsic or isinstance(bound, Value) and getattr(function, "__func__", None) is Value.with_components:
            def callback(arg):
                if isinstance(arg, (FunctionType, _Lambda, _Function)):
                    return lambda *a, **k: self.call(arg, list(a), k)
                return arg
            return function(*(callback(arg) for arg in args),
                            **{key: callback(arg) for key, arg in kwargs.items()})
        if module == "math" and all(isinstance(arg, (int, float)) for arg in args):
            return function(*args, **kwargs)
        if function in (builtins.int, builtins.float, builtins.len, builtins.tuple):
            if function in (builtins.int, builtins.float):
                if not all(type(arg) in (int, float, bool, str) for arg in args):
                    self.fail("Compile-time numeric conversion requires literal scalar operands.")
            elif len(args) != 1 or type(args[0]) not in (tuple, list, dict, str, range):
                self.fail("Compile-time collection operations require literal collections.")
            return function(*args, **kwargs)
        if function is builtins.range:
            if kwargs or not all(type(arg) is int for arg in args):
                self.fail("Device range bounds are only valid in a direct for ... in range(...) loop.")
            return range(*args)
        if function is builtins.reversed and len(args) == 1 and isinstance(args[0], (tuple, range)) and not kwargs:
            return tuple(reversed(args[0]))
        if isinstance(function, _Lambda):
            if kwargs or len(args) != len(function.parameters):
                self.fail("Lambda helper arguments must match its positional parameters.")
            previous = self.file, self.line, self.column, self.offset, self.indent, self.scope
            self.file, self.offset, self.indent = function.file, function.offset, function.indent
            self.scope = object()
            try:
                return self.expression(function.body, {**function.env, **dict(zip(function.parameters, args))})
            finally:
                self.file, self.line, self.column, self.offset, self.indent, self.scope = previous
        if isinstance(function, _Function):
            if function in self.stack:
                self.fail("Recursive device helper calls are unsupported.")
            parameters = function.definition.args
            positional = [arg.arg for arg in (*parameters.posonlyargs, *parameters.args)]
            if len(args) > len(positional):
                self.fail("Too many positional helper arguments.")
            values = dict(zip(positional, args))
            valid_keywords = {arg.arg for arg in (*parameters.args, *parameters.kwonlyargs)}
            for name, value in kwargs.items():
                if name not in valid_keywords or name in values:
                    self.fail("Unknown, positional-only, or duplicate helper keyword argument.")
                values[name] = value
            for name in (*positional, *(arg.arg for arg in parameters.kwonlyargs)):
                if name not in values:
                    if name not in function.defaults:
                        self.fail(f"Missing helper argument {name!r}.")
                    values[name] = function.defaults[name]
            previous = self.file, self.line, self.column, self.offset, self.indent, self.scope
            self.file, self.offset, self.indent = function.file, function.offset, function.indent
            self.scope = object()
            self.stack.append(function)
            try:
                return self.execute(function.definition.body, {**function.env, **values},
                                    lambda env: None, lambda value, env: value)
            finally:
                self.stack.pop()
                self.file, self.line, self.column, self.offset, self.indent, self.scope = previous
        if not isinstance(function, FunctionType):
            self.fail("Calls must resolve to static Python helpers or registered shader intrinsics.")
        if function in self.stack:
            self.fail("Recursive device helper calls are unsupported.")
        try:
            lines, start = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            self.fail(f"Helper source is unavailable; define it in a Python source file: {error}")
        parsed = ast.parse(textwrap.dedent("".join(lines)))
        definitions = [item for item in parsed.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
            self.fail("Only ordinary, statically defined device helper functions are supported.")
        definition = definitions[0]
        try:
            parameters = inspect.signature(function).bind(*args, **kwargs)
        except TypeError as error:
            self.fail(str(error))
        parameters.apply_defaults()
        closure = inspect.getclosurevars(function)
        env = {**vars(builtins), **function.__globals__, **closure.nonlocals, **parameters.arguments}
        for name in function.__code__.co_varnames:
            if name not in parameters.arguments:
                env.pop(name, None)
        previous = self.file, self.line, self.column, getattr(self, "offset", 0), getattr(self, "indent", 0), self.scope
        self.file = function.__code__.co_filename
        self.offset = start - 1
        self.indent = len(lines[0]) - len(lines[0].lstrip())
        self.scope = object()
        self.stack.append(function)
        try:
            return self.execute(definition.body, env, lambda env: None, lambda value, env: value)
        finally:
            self.stack.pop()
            self.file, self.line, self.column, self.offset, self.indent, self.scope = previous

    def expression(self, node, env):
        self.line = self.offset + node.lineno
        self.column = self.indent + node.col_offset + 1
        return self.located(lambda: self._expression(node, env))

    def _expression(self, node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                self.fail(f"Unresolved or possibly unbound local name {node.id!r}.")
            return env[node.id]
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                self.fail("Private attributes and Python introspection are unsupported in device source.")
            owner = self.expression(node.value, env)
            if not isinstance(owner, (Value, Node, ModuleType, Type, Vertex, Entry, Pipeline)) and owner is not Type:
                self.fail("Dynamic attributes on arbitrary Python objects are unsupported.")
            return getattr(owner, node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.expression(item, env) for item in node.elts)
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    fields = self.expression(value, env)
                    if type(fields) is not dict:
                        self.fail("Mapping expansion requires a compile-time field dictionary.")
                    result.update(fields)
                else:
                    result[self.expression(key, env)] = self.expression(value, env)
            return result
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            values = []
            def expand(index, scope):
                if index == len(node.generators):
                    values.append(self.expression(node.elt, scope))
                    return
                generator = node.generators[index]
                iterable = self.expression(generator.iter, scope)
                if generator.is_async or type(iterable) not in (tuple, list, range):
                    self.fail("Comprehensions require compile-time literal collections/ranges.")
                for item in iterable:
                    nested = dict(scope)
                    self.assign(generator.target, item, nested)
                    included = True
                    for condition in generator.ifs:
                        test = self.expression(condition, nested)
                        if isinstance(test, Value):
                            self.fail("Device-dependent comprehension filters cannot determine a static collection shape.")
                        if not test:
                            included = False
                            break
                    if included:
                        expand(index + 1, nested)
            expand(0, dict(env))
            return tuple(values)
        if isinstance(node, ast.Subscript):
            owner = self.expression(node.value, env)
            if type(owner) not in (Value, tuple, list, dict, str, range):
                self.fail("Subscripts require device vectors or compile-time literal collections.")
            return owner[self.expression(node.slice, env)]
        if isinstance(node, ast.Lambda):
            if node.args.vararg or node.args.kwarg or node.args.defaults or node.args.kwonlyargs:
                self.fail("Lambda helpers require plain positional arguments.")
            parameters = tuple(arg.arg for arg in (*node.args.posonlyargs, *node.args.args))
            captured = frozenset(item.id for item in ast.walk(node.body)
                                 if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                                 and item.id not in parameters)
            return _Lambda(parameters, node.body, dict(env), captured, self.scope,
                           self.file, self.offset, self.indent)
        if isinstance(node, ast.Call):
            function = self.expression(node.func, env)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    value = self.expression(arg.value, env)
                    if not isinstance(value, tuple):
                        self.fail("Argument expansion requires a compile-time tuple.")
                    args.extend(value)
                else:
                    args.append(self.expression(arg, env))
            if any(keyword.arg is None for keyword in node.keywords):
                self.fail("Dynamic keyword expansion is unsupported.")
            kwargs = {keyword.arg: self.expression(keyword.value, env) for keyword in node.keywords}
            return self.call(function, args, kwargs)
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, env), self.expression(node.right, env)
            if any(type(value) not in (Value, int, float, bool, str, tuple) for value in (left, right)):
                self.fail("Binary operations require device values or literal scalar/tuple constants.")
            operation = {
                ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
            }.get(type(node.op))
            if isinstance(node.op, ast.Pow):
                return shader_math.pow(left, right) if isinstance(left, Value) or isinstance(right, Value) else left ** right
            if operation is None:
                self.fail(f"Unsupported binary operation {type(node.op).__name__}.")
            return operation(left, right)
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, env)
            if isinstance(node.op, ast.Not):
                if isinstance(value, Value):
                    if value.node.type != Type.BOOL:
                        self.fail("Device not requires a bool operand.")
                    return ~value
                return not value
            operation = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}.get(type(node.op))
            if operation is None:
                self.fail("Unsupported unary operation.")
            return operation(value)
        if isinstance(node, ast.IfExp):
            return self.choose(self.expression(node.test, env),
                               lambda: self.expression(node.body, env), lambda: self.expression(node.orelse, env))
        if isinstance(node, ast.BoolOp):
            def combine(index):
                value = self.expression(node.values[index], env)
                if index == len(node.values) - 1:
                    return value
                return self.choose(value, lambda: combine(index + 1), lambda: value) if isinstance(node.op, ast.And) else \
                    self.choose(value, lambda: value, lambda: combine(index + 1))
            return combine(0)
        if isinstance(node, ast.Compare):
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                          ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            def compare(index, left):
                right = self.expression(node.comparators[index], env)
                operation = operations.get(type(node.ops[index]))
                if operation is None:
                    self.fail("Only scalar numeric/Boolean comparisons are supported.")
                value = operation(left, right)
                return value if index == len(node.ops) - 1 else \
                    self.choose(value, lambda: compare(index + 1, right), lambda: False)
            return compare(0, self.expression(node.left, env))
        self.fail(f"Unsupported expression: {type(node).__name__}.")

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            self.check_capture(target.id, env)
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) and len(target.elts) == len(value):
            for child, item in zip(target.elts, value):
                self.assign(child, item, env)
        else:
            self.fail("Device assignments require local names; attribute/subscript mutation is unsupported.")

    def check_capture(self, name, env):
        if any(isinstance(value, (_Function, _Lambda)) and value.owner is self.scope
               and name in value.captured
               for value in env.values()):
            self.fail(f"Captured local {name!r} cannot be rebound; pass updated state as an explicit helper argument.")

    def execute(self, statements, env, finish, returned, broken=None, continued=None):
        return self.resolve(self.plan(statements, env, finish, returned, broken, continued))

    def plan(self, statements, env, finish, returned, broken=None, continued=None):
        return _Next(lambda: self.step(statements, env, finish, returned, broken, continued))

    def step(self, statements, env, finish, returned, broken=None, continued=None):
        if not statements:
            return finish(env)
        node, *rest = statements
        self.line, self.column = self.offset + node.lineno, self.indent + node.col_offset + 1
        return self.located(lambda: self._execute(node, rest, env, finish, returned, broken, continued))

    def _execute(self, node, rest, env, finish, returned, broken, continued):
        if isinstance(node, ast.Return):
            return returned(self.expression(node.value, env) if node.value else None, env)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                self.fail("Local annotations require an initializer.")
            value = self.expression(node.value, env)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                self.assign(target, value, env)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.args.vararg or node.args.kwarg:
                self.fail("Local helpers must be static definitions without decorators or variadic arguments.")
            positional = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
            defaults = {name: self.expression(value, env) for name, value in
                        zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults)}
            defaults.update({arg.arg: self.expression(value, env) for arg, value in
                             zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None})
            locals = set(positional) | {arg.arg for arg in node.args.kwonlyargs} | {
                item.id for statement in node.body for item in ast.walk(statement)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            }
            captured = frozenset(
                item.id for statement in node.body for item in ast.walk(statement)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                and item.id not in locals and item.id != node.name
            )
            helper = _Function(node, dict(env), defaults, self.file, self.offset, self.indent, captured, self.scope)
            helper.env[node.name] = helper
            env[node.name] = helper
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                self.fail("Augmented assignment requires a local name.")
            expression = ast.copy_location(ast.BinOp(node.target, node.op, node.value), node)
            self.assign(node.target, self.expression(expression, env), env)
        elif isinstance(node, ast.If):
            return self.choose(self.expression(node.test, env),
                               lambda: self.plan(node.body + rest, dict(env), finish, returned, broken, continued),
                               lambda: self.plan(node.orelse + rest, dict(env), finish, returned, broken, continued))
        elif isinstance(node, (ast.For, ast.While)):
            return self.loop(node, env, lambda values: self.plan(rest, values, finish, returned, broken, continued), returned)
        elif isinstance(node, ast.Break):
            if broken is None:
                self.fail("break requires an enclosing device loop.")
            return broken(env)
        elif isinstance(node, ast.Continue):
            if continued is None:
                self.fail("continue requires an enclosing device loop.")
            return continued(env)
        elif isinstance(node, ast.Pass) or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass
        else:
            self.fail(f"Unsupported statement: {type(node).__name__}; mutation, exceptions, and dynamic Python execution are not device operations.")
        return self.plan(rest, env, finish, returned, broken, continued)

    def loop(self, node, env, continuation, returned):
        if node.orelse:
            self.fail("Loop else clauses are unsupported; express the termination result explicitly.")
        is_for = isinstance(node, ast.For)
        if is_for and not (isinstance(node.iter, ast.Call) and
                           self.expression(node.iter.func, env) is builtins.range):
            iterable = self.expression(node.iter, env)
            if not isinstance(iterable, (tuple, range)):
                self.fail("Static metaprogramming iteration requires a compile-time tuple/range, not a dynamic iterable.")
            items = tuple(iterable)
            def iteration(index, values):
                if index == len(items):
                    return continuation(values)
                self.assign(node.target, items[index], values)
                advance = lambda values: _Next(lambda: iteration(index + 1, values))
                return self.plan(node.body, values, advance, returned, continuation, advance)
            return _Next(lambda: iteration(0, dict(env)))
        self.structured = True
        parent = self.builder
        index = limit = step = None
        if is_for:
            if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or \
                    self.expression(node.iter.func, env) is not builtins.range or node.iter.keywords:
                self.fail("Device for loops require a local induction name and range().")
            args = [self.expression(arg, env) for arg in node.iter.args]
            if not 1 <= len(args) <= 3:
                self.fail("range requires one to three scalar bounds.")
            lower, upper, stride = (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
            if type(stride) is not int or not stride or not -(2**31) <= stride < 2**31:
                self.fail("range step must be a nonzero compile-time signed 32-bit integer.")
            bounds = [self.lift(value, Type.I32) for value in (lower, upper, stride)]
            index, limit, step = [parent.raw("arith.index_cast", [parent.emit(value.node)], ["index"])[0] for value in bounds]
        assigned = {item.id for statement in node.body for item in ast.walk(statement)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}
        if is_for:
            self.check_capture(node.target.id, env)
            env = dict(env)
            if node.target.id not in env and type(lower) is int and type(upper) is int and \
                    (lower < upper if stride > 0 else lower > upper):
                env[node.target.id] = self.lift(lower, Type.I32)
            assigned.add(node.target.id)
        names = [name for name in env if name in assigned]
        initial = {name: self.lift(env[name]) for name in names}
        types = [value.node.type for value in initial.values()]
        initial_refs = [parent.emit(value.node) for value in initial.values()]
        if is_for:
            initial_refs.append(index)
        argument_types = [value.type for value in initial_refs]
        arguments = [{"name": self.name(), "type": typ} for typ in argument_types]
        after = _Builder(self, arguments, parent.cache)
        local = dict(env)
        for name, arg, typ in zip(names, arguments, types):
            local[name] = after.bind(_Ref(arg["name"], arg["type"]), typ)
        if is_for:
            iv = _Ref(arguments[-1]["name"], "index")
            converted, = after.raw("arith.index_cast", [iv], ["i32"])
            local[node.target.id] = after.bind(converted, Type.I32)
        def exit(values, mode=0, value=None):
            return _LoopExit({name: self.lift(values[name], typ) for name, typ in zip(names, types)}, mode, value)
        try:
            self.builder = after
            outcome = self.execute(node.body, local, exit, lambda value, values: exit(values, 2, value),
                                   lambda values: exit(values, 1), lambda values: exit(values))
            leaves, rebuild_return = self.pair(outcome.returned, outcome.returned)
            return_types = [value.node.type for value, _ in leaves]
            yielded = [after.emit(outcome.values[name].node) for name in names]
            if is_for:
                advanced, = after.raw("arith.addi", [iv, step], ["index"])
                yielded.append(advanced)
                distance, = after.raw("arith.subi", [limit, iv] if stride > 0 else [iv, limit], ["index"])
                zero_index, = after.raw("arith.constant", types=["index"],
                                       attributes={"value": {"kind": "literal", "type": "index", "value": 0}})
                magnitude = step if stride > 0 else after.raw("arith.subi", [zero_index, step], ["index"])[0]
                ref, = after.raw("arith.cmpi", [distance, magnitude], ["bool"],
                                 {"predicate": {"kind": "cmpi", "value": "ugt"}})
                can_advance = after.bind(ref, Type.BOOL)
                mode = self.lift(outcome.mode, Type.I32)
                # Stop before a signed 32-bit induction wrap on SPIR-V's
                # 32-bit index lowering; normal range completion is not break.
                outcome.mode = select((mode == 0) & ~can_advance, self.lift(4, Type.I32), mode)
            yielded.append(after.emit(self.lift(outcome.mode, Type.I32).node))
            yielded.extend(after.emit(value.node) for value, _ in leaves)
            after.raw("scf.yield", yielded)
        finally:
            self.builder = parent
        mode0 = parent.emit(self.lift(0, Type.I32).node)
        initial_refs.append(mode0)
        zero = self.zero(outcome.returned)
        zero_leaves, _ = self.pair(zero, zero)
        initial_refs.extend(parent.emit(value.node) for value, _ in zero_leaves)
        tail_types = ["i32", *(typ.schema_name for typ in return_types)]
        after.block["arguments"].extend({"name": self.name(), "type": typ} for typ in tail_types)
        before_args = [{"name": self.name(), "type": ref.type} for ref in initial_refs]
        before = _Builder(self, before_args, parent.cache)
        condition_env = dict(env)
        for name, arg, typ in zip(names, before_args, types):
            condition_env[name] = before.bind(_Ref(arg["name"], arg["type"]), typ)
        mode_index = len(names) + int(is_for)
        mode = before.bind(_Ref(before_args[mode_index]["name"], "i32"), Type.I32)
        try:
            self.builder = before
            if is_for:
                ref, = before.raw("arith.cmpi", [_Ref(before_args[len(names)]["name"], "index"), limit],
                                  ["bool"], {"predicate": {"kind": "cmpi", "value": "slt" if stride > 0 else "sgt"}})
                condition = before.bind(ref, Type.BOOL) & (mode == 0)
            else:
                condition = self.choose(mode == 0, lambda: self.expression(node.test, condition_env), lambda: False)
                condition = self.lift(condition, Type.BOOL)
            before.raw("scf.condition", [before.emit(condition.node), *[_Ref(arg["name"], arg["type"]) for arg in before_args]])
        finally:
            self.builder = parent
        refs = parent.raw("scf.while", initial_refs, [ref.type for ref in initial_refs], regions=[before, after])
        final_env = dict(env)
        for name, ref, typ in zip(names, refs, types):
            final_env[name] = parent.bind(ref, typ)
        mode = parent.bind(refs[mode_index], Type.I32)
        if outcome.returned is None:
            return continuation(final_env)
        result = rebuild_return([parent.bind(ref, typ) for ref, typ in zip(refs[mode_index + 1:], return_types)])
        return self.choose(mode == 2, lambda: returned(result, final_env), lambda: continuation(final_env))

    def module(self, name, kind, outputs, interfaces):
        for node, interface in zip(outputs, interfaces):
            ref = self.root.emit(node)
            if interface.builtin == "position":
                self.root.raw("gfx.position", [ref])
            else:
                self.root.raw("gfx.output", [ref], attributes={
                    "location": {"kind": "literal", "type": "i32", "value": interface.location},
                    "name": {"kind": "string", "value": interface.name},
                    "interpolation": {"kind": "interpolation", "value": interface.interpolation},
                })
        self.root.raw("func.return")
        return {"schema_version": 2, "functions": [{
            "name": name, "stage": kind, "results": [],
            "body": {"blocks": [self.root.block]}, **self.location(),
        }]}


@dataclass(frozen=True, eq=False)
class _Lambda:
    parameters: tuple[str, ...]
    body: ast.expr
    env: dict
    captured: frozenset[str]
    owner: object
    file: str
    offset: int
    indent: int


@dataclass(frozen=True, eq=False)
class _Function:
    definition: ast.FunctionDef
    env: dict
    defaults: dict
    file: str
    offset: int
    indent: int
    captured: frozenset[str]
    owner: object
