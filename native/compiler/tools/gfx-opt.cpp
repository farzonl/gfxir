#include "Gfx/Gfx.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::gfx::registerGfxDialects(registry);
  mlir::registerTransformsPasses();
  mlir::gfx::registerGfxPasses();
  mlir::gfx::registerGfxPipelines();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "gfx compiler optimizer\n", registry));
}
