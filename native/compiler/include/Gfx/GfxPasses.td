#ifndef GFX_PASSES_TD
#define GFX_PASSES_TD
include "mlir/Pass/PassBase.td"

def GfxToSPIRV : Pass<"convert-gfx-to-spirv", "::mlir::ModuleOp"> {
  let summary = "Lower verified graphics interfaces and upstream arithmetic to SPIR-V";
  let constructor = "::mlir::gfx::createGfxToSPIRVPass()";
  let dependentDialects = ["::mlir::spirv::SPIRVDialect", "::mlir::vector::VectorDialect"];
}

def GfxToLLVM : Pass<"convert-gfx-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower graphics interfaces to upstream LLVM DirectX intrinsics";
  let constructor = "::mlir::gfx::createGfxToLLVMPass()";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def GfxVerify : Pass<"gfx-verify", "::mlir::ModuleOp"> {
  let summary = "Verify shader entry points and stage interface contracts";
  let constructor = "::mlir::gfx::createGfxVerifyPass()";
}
#endif
