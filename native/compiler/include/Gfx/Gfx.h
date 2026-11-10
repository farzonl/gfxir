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
#include <array>
#include <optional>
#include <variant>
#include <vector>

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
  ParameterBlockAttr parameters;
};

struct ParameterFieldLayout {
  ParameterFieldAttr field;
  std::string name;
  Type type;
  uint32_t offset = 0;
  uint32_t size = 0;
  uint32_t alignment = 1;
  uint32_t paddingAfter = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
  std::vector<ParameterFieldLayout> nestedFields;
};
struct ParameterLayout {
  SmallVector<ParameterFieldLayout> fields;
  uint32_t size = 0;
  uint32_t alignment = 16;
};
ParameterLayout parameterLayout(ParameterBlockAttr block);
std::vector<ParameterFieldLayout> layoutStructMembers(StructType type);
LogicalResult getStd430Layout(Type type, uint32_t &size, uint32_t &alignment);
llvm::json::Object serializeParameters(ParameterBlockAttr block, Stage stage);
LogicalResult emitCHeader(const ParameterLayout &layout, llvm::raw_ostream &os,
                          llvm::StringRef structName);
LogicalResult emitCHeader(const ShaderInterface &reflection,
                          llvm::raw_ostream &os);

struct TargetProfile {
  uint32_t spirv_version = 0x10000; // 1.0 default (0xMMmm00)
  uint32_t shader_model = 0x60;     // 6.0 default (0xMm)
  llvm::SmallVector<std::string, 8> required_capabilities;
  std::array<uint32_t, 3> max_workgroup_size = {1024, 1024, 64};
  uint32_t max_workgroup_invocations = 1024;
  uint32_t max_shared_memory_bytes = 32768;

  bool supportsCapability(llvm::StringRef cap) const;
  llvm::LogicalResult verifyWorkgroupSize(llvm::ArrayRef<int32_t> dims,
                                          mlir::Location loc) const;
  llvm::LogicalResult verifySharedMemory(uint32_t bytes,
                                         mlir::Location loc) const;
  llvm::LogicalResult verifyCapability(llvm::StringRef cap,
                                       mlir::Location loc) const;

  static TargetProfile getDefaultVulkan();
  static TargetProfile getDefaultDirectX();
  static llvm::Expected<TargetProfile> fromJson(const llvm::json::Object &object);
  llvm::json::Object toJson() const;

  static std::optional<uint32_t> parseSpirvVersion(llvm::StringRef str);
  static std::optional<uint32_t> parseShaderModel(llvm::StringRef str);
  static std::string formatSpirvVersion(uint32_t version);
  static std::string formatShaderModel(uint32_t model);
};

/// Returns the natural alignment in bytes for a supported type according to std430 rules.
/// Returns 0 for unsupported or non-concrete types.
uint32_t getNaturalAlignment(Type type);

/// Returns the size in bytes for a supported scalar or vector type.
uint32_t getTypeSize(Type type);

/// Returns true if the type is a legal pointee type for !gfx.ptr.
bool isValidPointeeType(Type type);

std::unique_ptr<Pass> createGfxToSPIRVPass(const TargetProfile &profile = TargetProfile());
std::unique_ptr<Pass> createGfxToLLVMPass(const TargetProfile &profile = TargetProfile());
std::unique_ptr<Pass> createGfxVerifyPass();
void buildDirectXPipeline(OpPassManager &manager, const TargetProfile &profile = TargetProfile());
void registerGfxPipelines();
#define GEN_PASS_REGISTRATION
#include "Gfx/GfxPasses.h.inc"

void registerGfxDialects(DialectRegistry &registry);
LogicalResult verifyShaderModule(ModuleOp module,
                                 const TargetProfile &profile = TargetProfile());
Type parseShaderType(MLIRContext &context, StringRef name);
bool isShaderSymbolName(StringRef name);
Value buildPortablePow(OpBuilder &builder, Location location, Value base,
                       Value exponent);
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
