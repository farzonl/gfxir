// REQUIRES: spirv-val
// RUN: %split-file %s %t
// RUN: %gfx-opt %t/fragment_texture.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-FRAG
// RUN: %gfx-compile %t/fragment_texture.mlir --input-format mlir --target vulkan --output %t/frag.spv
// RUN: %spirv-val --target-env vulkan1.0 %t/frag.spv
// RUN: %gfx-opt %t/compute_texture.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-COMP
// RUN: %gfx-compile %t/compute_texture.mlir --input-format mlir --target vulkan --output %t/comp.spv
// RUN: %spirv-val --target-env vulkan1.0 %t/comp.spv

// CHECK-FRAG-LABEL: spirv.module Logical GLSL450
// CHECK-FRAG-DAG: spirv.GlobalVariable @{{.*}} bind(0, 0) : !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>, UniformConstant>
// CHECK-FRAG-DAG: spirv.GlobalVariable @{{.*}} bind(0, 1) : !spirv.ptr<!spirv.sampler, UniformConstant>
// CHECK-FRAG-LABEL: spirv.func @frag_tex_main()
// CHECK-FRAG: spirv.SampledImage
// CHECK-FRAG: spirv.ImageSampleImplicitLod
// CHECK-FRAG: spirv.ImageSampleExplicitLod %{{.*}}, %{{.*}} ["Lod"], %{{.*}}
// CHECK-FRAG: spirv.ImageFetch %{{.*}}, %{{.*}} ["Lod"], %{{.*}}
// CHECK-FRAG: spirv.EntryPoint "Fragment" @frag_tex_main

// CHECK-COMP-LABEL: spirv.module Logical GLSL450
// CHECK-COMP-DAG: spirv.GlobalVariable @{{.*}} bind(0, 0) : !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>, UniformConstant>
// CHECK-COMP-DAG: spirv.GlobalVariable @{{.*}} bind(0, 1) : !spirv.ptr<!spirv.sampler, UniformConstant>
// CHECK-COMP-DAG: spirv.GlobalVariable @{{.*}} bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0]), Block>, StorageBuffer>
// CHECK-COMP-LABEL: spirv.func @comp_tex_main()
// CHECK-COMP: spirv.SampledImage
// CHECK-COMP: spirv.ImageSampleExplicitLod %{{.*}}, %{{.*}} ["Lod"], %{{.*}}
// CHECK-COMP: spirv.ImageFetch %{{.*}}, %{{.*}} ["Lod"], %{{.*}}
// CHECK-COMP: spirv.Store "StorageBuffer"
// CHECK-COMP: spirv.EntryPoint "GLCompute" @comp_tex_main

//--- fragment_texture.mlir
module {
  func.func @frag_tex_main() attributes {gfx.stage = #gfx.stage<fragment>} {
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

//--- compute_texture.mlir
module {
  func.func @comp_tex_main(
      %dst_buf: !gfx.ptr<vector<4xf32>, global> {gfx.binding = 2 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 4, 4, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %gx = vector.extract %gid[0] : i32 from vector<3xi32>
    %gy = vector.extract %gid[1] : i32 from vector<3xi32>
    %tex = gfx.texture_binding 0, 0, "src_tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "src_samp" : !gfx.handle<sampler>
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %lod_f = arith.constant 0.0 : f32
    %s = gfx.texture_sample_level %tex, %samp, %uv, %lod_f : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
    %coord = vector.from_elements %gx, %gy : vector<2xi32>
    %lod_i = arith.constant 0 : i32
    %f = gfx.texture_load %tex, %coord, %lod_i : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>
    %sum = arith.addf %s, %f : vector<4xf32>
    %ptr = gfx.ptr_offset %dst_buf, %gx : (!gfx.ptr<vector<4xf32>, global>, i32) -> !gfx.ptr<vector<4xf32>, global>
    gfx.store %sum, %ptr : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}
