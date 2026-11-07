// RUN: %split-file %s %t
// RUN: not %gfx-opt %t/recursive.mlir --convert-gfx-to-spirv 2>&1 | %FileCheck %s --check-prefix=RECURSIVE
// RUN: not %gfx-opt %t/recursive.mlir --gfx-to-directx 2>&1 | %FileCheck %s --check-prefix=RECURSIVE
// RUN: not %gfx-opt %t/external.mlir --convert-gfx-to-spirv 2>&1 | %FileCheck %s --check-prefix=EXTERNAL
// RUN: not %gfx-opt %t/multiple.mlir --convert-gfx-to-spirv 2>&1 | %FileCheck %s --check-prefix=MULTIPLE
// RECURSIVE: recursive shader calls
// EXTERNAL: structured body
// MULTIPLE: at most one result

//--- recursive.mlir
module {
  func.func private @helper() -> f32 {
    %value = func.call @helper() : () -> f32
    return %value : f32
  }
  func.func @entry() attributes {gfx.stage = #gfx.stage<fragment>} {
    %value = func.call @helper() : () -> f32
    %color = vector.broadcast %value : f32 to vector<4xf32>
    gfx.output %color 0 "color" <perspective> : vector<4xf32>
    return
  }
}
//--- external.mlir
module {
  func.func private @helper() -> f32
  func.func @entry() attributes {gfx.stage = #gfx.stage<fragment>} {
    %value = func.call @helper() : () -> f32
    %color = vector.broadcast %value : f32 to vector<4xf32>
    gfx.output %color 0 "color" <perspective> : vector<4xf32>
    return
  }
}
//--- multiple.mlir
module {
  func.func private @helper() -> (f32, f32) {
    %zero = arith.constant 0.0 : f32
    return %zero, %zero : f32, f32
  }
  func.func @entry() attributes {gfx.stage = #gfx.stage<fragment>} {
    %value, %unused = func.call @helper() : () -> (f32, f32)
    %color = vector.broadcast %value : f32 to vector<4xf32>
    gfx.output %color 0 "color" <perspective> : vector<4xf32>
    return
  }
}
