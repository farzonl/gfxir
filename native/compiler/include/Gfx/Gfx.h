#ifndef GFX_COMPILER_H
#define GFX_COMPILER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <variant>

#include "Gfx/GfxDialect.h.inc"
#include "Gfx/GfxEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Gfx/GfxAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "Gfx/GfxTypes.h.inc"
#define GET_OP_CLASSES
#include "Gfx/GfxOps.h.inc"

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir::gfx {
struct PositionBinding {};
using InterfaceBinding = std::variant<Builtin, int32_t, PositionBinding>;

struct InterfaceElement {
  std::string name;
  Type type;
  InterfaceBinding binding;
  Interpolation interpolation;
};

struct ShaderInterface {
  std::string name;
  Stage stage;
  SmallVector<InterfaceElement> inputs;
  SmallVector<InterfaceElement> outputs;
};

std::unique_ptr<Pass> createGfxToSPIRVPass();
std::unique_ptr<Pass> createGfxToLLVMPass();
std::unique_ptr<Pass> createGfxVerifyPass();
void buildDirectXPipeline(OpPassManager &manager);
void registerGfxPipelines();
#define GEN_PASS_REGISTRATION
#include "Gfx/GfxPasses.h.inc"

void registerGfxDialects(DialectRegistry &registry);
LogicalResult verifyShaderModule(ModuleOp module);
Type parseShaderType(MLIRContext &context, StringRef name);
bool isShaderSymbolName(StringRef name);
OwningOpRef<ModuleOp> importShader(MLIRContext &context,
                                   const llvm::json::Object &source);
OwningOpRef<ModuleOp> importStructuredShader(MLIRContext &context,
                                             const llvm::json::Object &source);
ShaderInterface reflectShader(ModuleOp module);
llvm::json::Object serializeReflection(const ShaderInterface &reflection);
std::unique_ptr<llvm::Module> exportDirectX(ModuleOp module,
                                            llvm::LLVMContext &context,
                                            const ShaderInterface &reflection);
} // namespace mlir::gfx
#endif
