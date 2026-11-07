#ifndef GFX_DIALECT_TD
#define GFX_DIALECT_TD
include "mlir/IR/OpBase.td"

def Gfx_Dialect : Dialect {
  let name = "gfx";
  let cppNamespace = "::mlir::gfx";
  let summary = "Target-independent graphics and compute interfaces";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 1;
}

class Gfx_Op<string mnemonic, list<Trait> traits = []>
    : Op<Gfx_Dialect, mnemonic, traits>;
#endif
