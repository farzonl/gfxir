// RUN: %gfx-opt %s --gfx-verify | %FileCheck %s --check-prefix=VERIFY
// RUN: %gfx-opt %s --convert-gfx-to-spirv | %FileCheck %s --check-prefix=SPIRV
// RUN: %gfx-opt %s --gfx-to-directx | %FileCheck %s --check-prefix=DXIL

module {
  func.func @vertex_main() attributes {gfx.stage = #gfx.stage<vertex>} {
    // VERIFY: gfx.input 0 "position" <perspective> : vector<4xf32>
    // VERIFY: gfx.input 1 "uv" <perspective> : vector<2xf32>
    %pos = gfx.input 0 "position" <perspective> : vector<4xf32>
    %uv = gfx.input 1 "uv" <perspective> : vector<2xf32>

    // SPIRV: spirv.GlobalVariable @interface_0 {location = 0 : i32} : !spirv.ptr<vector<4xf32>, Input>
    // SPIRV: spirv.GlobalVariable @interface_1 {location = 1 : i32} : !spirv.ptr<vector<2xf32>, Input>
    // SPIRV: spirv.GlobalVariable @interface_2 built_in("Position") : !spirv.ptr<vector<4xf32>, Output>
    // SPIRV: spirv.GlobalVariable @interface_3 {location = 0 : i32} : !spirv.ptr<vector<2xf32>, Output>
    // SPIRV: spirv.EntryPoint "Vertex" @vertex_main, @interface_0, @interface_1, @interface_2, @interface_3

    // DXIL: %[[LOC1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // DXIL: %[[LOC0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // DXIL: llvm.call_intrinsic "llvm.dx.load.input"(%[[LOC0]], %[[LOC0]],
    // DXIL: llvm.call_intrinsic "llvm.dx.load.input"(%[[LOC1]], %[[LOC0]],
    gfx.position %pos : vector<4xf32>
    gfx.output %uv 0 "uv" <perspective> : vector<2xf32>
    return
  }
}
