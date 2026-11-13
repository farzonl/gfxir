// REQUIRES: llc, dxv
// RUN: %split-file %s %t
// RUN: %gfx-opt %t/fragment_texture.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-FRAG-MLIR
// RUN: %gfx-compile %t/fragment_texture.mlir --input-format mlir --emit-llvm -S | %FileCheck %s --check-prefix=CHECK-FRAG-LLVM
// RUN: %gfx-compile %t/fragment_texture.mlir --input-format mlir --target directx --output %t/frag.ll
// RUN: %llc -O0 -filetype=obj %t/frag.ll -o %t/frag.unsigned.dxil
// RUN: %dxv %t/frag.unsigned.dxil -o %t/frag.dxil
// RUN: %gfx-opt %t/compute_mixed_resources.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-COMP-MLIR
// RUN: %gfx-compile %t/compute_mixed_resources.mlir --input-format mlir --emit-llvm -S | %FileCheck %s --check-prefix=CHECK-COMP-LLVM
// RUN: %gfx-compile %t/compute_mixed_resources.mlir --input-format mlir --target directx --output %t/comp.ll
// RUN: %llc -O0 -filetype=obj %t/comp.ll -o %t/comp.unsigned.dxil
// RUN: %dxv %t/comp.unsigned.dxil -o %t/comp.dxil

// CHECK-FRAG-MLIR: module attributes {llvm.target_triple = "dxil-pc-shadermodel6.6-pixel"}
// CHECK-FRAG-MLIR-LABEL: llvm.func @frag_dx_main()
// CHECK-FRAG-MLIR-DAG: llvm.call_intrinsic "llvm.dx.resource.handlefrombinding"{{.*}}!llvm.target<"dx.Texture", vector<4xf32>, 0, 0, 0, 2>
// CHECK-FRAG-MLIR-DAG: llvm.call_intrinsic "llvm.dx.resource.handlefrombinding"{{.*}}!llvm.target<"dx.Sampler", 0>
// CHECK-FRAG-MLIR: llvm.call_intrinsic "llvm.dx.resource.sample"
// CHECK-FRAG-MLIR: llvm.call_intrinsic "llvm.dx.resource.samplelevel"
// CHECK-FRAG-MLIR: llvm.call_intrinsic "llvm.dx.resource.load.level"

// CHECK-FRAG-LLVM: target triple = "dxil-pc-shadermodel6.6-pixel"
// CHECK-FRAG-LLVM-LABEL: define void @frag_dx_main()
// CHECK-FRAG-LLVM-DAG: call target("dx.Texture", <4 x float>, 0, 0, 0, 2) @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(i32 0, i32 0, i32 1, i32 0
// CHECK-FRAG-LLVM-DAG: call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(i32 0, i32 0, i32 1, i32 0
// CHECK-FRAG-LLVM: call <4 x float> @llvm.dx.resource.sample
// CHECK-FRAG-LLVM: call <4 x float> @llvm.dx.resource.samplelevel
// CHECK-FRAG-LLVM: call <4 x float> @llvm.dx.resource.load.level
// CHECK-FRAG-LLVM: !dx.rootsignatures
// CHECK-FRAG-LLVM-DAG: !"SRV", i32 1, i32 0, i32 0
// CHECK-FRAG-LLVM-DAG: !"Sampler", i32 1, i32 0, i32 0

// CHECK-COMP-MLIR-LABEL: llvm.func @comp_dx_main()
// CHECK-COMP-MLIR-SAME: "hlsl.shader", "compute"
// CHECK-COMP-MLIR: llvm.call_intrinsic "llvm.dx.resource.samplelevel"
// CHECK-COMP-MLIR: llvm.call_intrinsic "llvm.dx.resource.load.level"
// CHECK-COMP-MLIR: llvm.store

// CHECK-COMP-LLVM: target triple = "dxil-pc-shadermodel6.6-compute"
// CHECK-COMP-LLVM-LABEL: define void @comp_dx_main()
// Unified SRV ordering: scale_buf at (0,0) gets t0, src_tex at (0,1) gets t1, sampler at (0,2) gets s0, out_buf at (0,3) gets u0
// CHECK-COMP-LLVM-DAG: call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 0, i32 1, i32 0
// CHECK-COMP-LLVM-DAG: call target("dx.Texture", <4 x float>, 0, 0, 0, 2) @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(i32 0, i32 1, i32 1, i32 0
// CHECK-COMP-LLVM-DAG: call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(i32 0, i32 0, i32 1, i32 0
// CHECK-COMP-LLVM-DAG: call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0
// CHECK-COMP-LLVM: !dx.rootsignatures
// CHECK-COMP-LLVM-DAG: !"SRV", i32 2, i32 0, i32 0
// CHECK-COMP-LLVM-DAG: !"UAV", i32 1, i32 0, i32 0
// CHECK-COMP-LLVM-DAG: !"Sampler", i32 1, i32 0, i32 0

//--- fragment_texture.mlir
module {
  func.func @frag_dx_main() attributes {gfx.stage = #gfx.stage<fragment>} {
    %uv = gfx.input 0 "uv" <perspective> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "albedo" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "linear" : !gfx.handle<sampler>
    %implicit = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    %lod_f = arith.constant 0.0 : f32
    %explicit = gfx.texture_sample_level %tex, %samp, %uv, %lod_f : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
    %coord = arith.constant dense<[0, 0]> : vector<2xi32>
    %lod_i = arith.constant 0 : i32
    %fetched = gfx.texture_load %tex, %coord, %lod_i : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>
    %tmp = arith.addf %implicit, %explicit : vector<4xf32>
    %out = arith.addf %tmp, %fetched : vector<4xf32>
    gfx.output %out 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- compute_mixed_resources.mlir
module {
  func.func @comp_dx_main(
      %scale_buf: !gfx.ptr<f32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>},
      %out_buf: !gfx.ptr<vector<4xf32>, global> {gfx.binding = 3 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 4, 4, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %gx = vector.extract %gid[0] : i32 from vector<3xi32>
    %gy = vector.extract %gid[1] : i32 from vector<3xi32>
    %tex = gfx.texture_binding 0, 1, "src_tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 2, "src_samp" : !gfx.handle<sampler>
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %lod_f = arith.constant 0.0 : f32
    %sampled = gfx.texture_sample_level %tex, %samp, %uv, %lod_f : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
    %coord = vector.from_elements %gx, %gy : vector<2xi32>
    %lod_i = arith.constant 0 : i32
    %fetched = gfx.texture_load %tex, %coord, %lod_i : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>
    %s_ptr = gfx.ptr_offset %scale_buf, %gx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %scale = gfx.load %s_ptr : (!gfx.ptr<f32, global>) -> f32
    %scale_v = vector.broadcast %scale : f32 to vector<4xf32>
    %sum = arith.addf %sampled, %fetched : vector<4xf32>
    %scaled = arith.mulf %sum, %scale_v : vector<4xf32>
    %d_ptr = gfx.ptr_offset %out_buf, %gx : (!gfx.ptr<vector<4xf32>, global>, i32) -> !gfx.ptr<vector<4xf32>, global>
    gfx.store %scaled, %d_ptr : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}
