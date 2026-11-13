// RUN: %split-file %s %t
// RUN: %gfx-opt %t/fragment_sample.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-FRAG
// RUN: %gfx-opt %t/vertex_sample_level.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VERT
// RUN: %gfx-opt %t/compute_sample_and_load.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-COMP
// RUN: %gfx-compile %t/fragment_sample.mlir --input-format mlir --target vulkan --output %t/frag.spv --reflection %t/frag.json
// RUN: %FileCheck %s --input-file=%t/frag.json --check-prefix=CHECK-REFL
// RUN: not %gfx-opt %t/err_sample_in_vertex.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-VERT-SAMPLE
// RUN: not %gfx-opt %t/err_sample_in_compute.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-COMP-SAMPLE
// RUN: not %gfx-opt %t/err_sample_in_helper_from_compute.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-HELPER-SAMPLE
// RUN: not %gfx-opt %t/err_conflicting_tex_sampler_slot.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-CONFLICT-TEX-SAMP
// RUN: not %gfx-opt %t/err_conflicting_tex_buffer_slot.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-CONFLICT-TEX-BUF
// RUN: not %gfx-opt %t/err_tex_binding_wrong_handle.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-TEX-HANDLE
// RUN: not %gfx-opt %t/err_sampler_binding_wrong_handle.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SAMP-HANDLE
// RUN: not %gfx-opt %t/err_sample_swapped_handles.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SWAPPED-HANDLES
// RUN: not %gfx-opt %t/err_load_float_coords.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-LOAD-COORDS

// CHECK-FRAG-LABEL: func.func @frag_main
// CHECK-FRAG: gfx.texture_binding 0, 0, "albedo_tex" : !gfx.handle<texture_2d>
// CHECK-FRAG: gfx.sampler_binding 0, 1, "linear_sampler" : !gfx.handle<sampler>
// CHECK-FRAG: gfx.texture_sample %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
// CHECK-FRAG: gfx.texture_load %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>

// CHECK-VERT-LABEL: func.func @vert_main
// CHECK-VERT: gfx.texture_binding 0, 0, "disp_tex" : !gfx.handle<texture_2d>
// CHECK-VERT: gfx.sampler_binding 0, 1, "disp_sampler" : !gfx.handle<sampler>
// CHECK-VERT: gfx.texture_sample_level %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>

// CHECK-COMP-LABEL: func.func @comp_main
// CHECK-COMP: gfx.texture_binding 0, 0, "src_tex" : !gfx.handle<texture_2d>
// CHECK-COMP: gfx.sampler_binding 0, 1, "src_sampler" : !gfx.handle<sampler>
// CHECK-COMP: gfx.texture_sample_level %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
// CHECK-COMP: gfx.texture_load %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>

// CHECK-REFL: "samplers":[{"binding":1,"name":"linear_sampler","set":0}]
// CHECK-REFL: "textures":[{"binding":0,"dimension":"2d","name":"albedo_tex","sample_type":"f32x4","set":0}]

// ERR-VERT-SAMPLE: implicit-LOD texture_sample is only allowed in fragment shaders; use texture_sample_level
// ERR-COMP-SAMPLE: implicit-LOD texture_sample is only allowed in fragment shaders; use texture_sample_level
// ERR-HELPER-SAMPLE: implicit-LOD texture_sample is only allowed in fragment shaders; use texture_sample_level
// ERR-CONFLICT-TEX-SAMP: conflicting descriptor declarations for set 0, binding 0
// ERR-CONFLICT-TEX-BUF: conflicting descriptor declarations for set 0, binding 1
// ERR-TEX-HANDLE: texture_binding must produce !gfx.handle<texture_2d>
// ERR-SAMP-HANDLE: sampler_binding must produce !gfx.handle<sampler>
// ERR-SWAPPED-HANDLES: texture_sample texture operand must be !gfx.handle<texture_2d>
// ERR-LOAD-COORDS: texture_load coordinate operand must be vector<2xi32>

//--- fragment_sample.mlir
module {
  func.func @frag_main() attributes {gfx.stage = #gfx.stage<fragment>} {
    %uv = gfx.input 0 "uv" <perspective> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "albedo_tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "linear_sampler" : !gfx.handle<sampler>
    %sampled = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    %coord = arith.constant dense<[0, 0]> : vector<2xi32>
    %lod = arith.constant 0 : i32
    %fetched = gfx.texture_load %tex, %coord, %lod : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>
    %sum = arith.addf %sampled, %fetched : vector<4xf32>
    gfx.output %sum 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- vertex_sample_level.mlir
module {
  func.func @vert_main() attributes {gfx.stage = #gfx.stage<vertex>} {
    %uv = gfx.input 0 "uv" <perspective> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "disp_tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "disp_sampler" : !gfx.handle<sampler>
    %lod = arith.constant 0.0 : f32
    %pos = gfx.texture_sample_level %tex, %samp, %uv, %lod : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}

//--- compute_sample_and_load.mlir
module {
  func.func @comp_main(
      %out: !gfx.ptr<vector<4xf32>, global> {gfx.binding = 2 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    %tex = gfx.texture_binding 0, 0, "src_tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "src_sampler" : !gfx.handle<sampler>
    %uv = arith.constant dense<[0.25, 0.75]> : vector<2xf32>
    %lod_f = arith.constant 0.0 : f32
    %sampled = gfx.texture_sample_level %tex, %samp, %uv, %lod_f : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>, f32) -> vector<4xf32>
    %coord = arith.constant dense<[1, 0]> : vector<2xi32>
    %lod_i = arith.constant 0 : i32
    %fetched = gfx.texture_load %tex, %coord, %lod_i : (!gfx.handle<texture_2d>, vector<2xi32>, i32) -> vector<4xf32>
    %combined = arith.addf %sampled, %fetched : vector<4xf32>
    %c0 = arith.constant 0 : i32
    %dst = gfx.ptr_offset %out, %c0 : (!gfx.ptr<vector<4xf32>, global>, i32) -> !gfx.ptr<vector<4xf32>, global>
    gfx.store %combined, %dst : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}

//--- err_sample_in_vertex.mlir
module {
  func.func @bad_vert() attributes {gfx.stage = #gfx.stage<vertex>} {
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "samp" : !gfx.handle<sampler>
    %pos = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}

//--- err_sample_in_compute.mlir
module {
  func.func @bad_comp() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "samp" : !gfx.handle<sampler>
    %val = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    return
  }
}

//--- err_sample_in_helper_from_compute.mlir
module {
  func.func private @helper(%uv: vector<2xf32>) -> vector<4xf32> {
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "samp" : !gfx.handle<sampler>
    %val = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    return %val : vector<4xf32>
  }
  func.func @bad_comp_caller() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %c = func.call @helper(%uv) : (vector<2xf32>) -> vector<4xf32>
    return
  }
}

//--- err_conflicting_tex_sampler_slot.mlir
module {
  func.func @conflict_main() attributes {gfx.stage = #gfx.stage<fragment>} {
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 0, "samp" : !gfx.handle<sampler>
    %col = gfx.texture_sample %tex, %samp, %uv : (!gfx.handle<texture_2d>, !gfx.handle<sampler>, vector<2xf32>) -> vector<4xf32>
    gfx.output %col 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_conflicting_tex_buffer_slot.mlir
module {
  func.func @conflict_buf_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    %buf = gfx.buffer_binding 0, 1, "buf", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    %tex = gfx.texture_binding 0, 1, "tex" : !gfx.handle<texture_2d>
    return
  }
}

//--- err_tex_binding_wrong_handle.mlir
module {
  func.func @wrong_tex_handle() attributes {gfx.stage = #gfx.stage<fragment>} {
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<sampler>
    %col = arith.constant dense<1.0> : vector<4xf32>
    gfx.output %col 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_sampler_binding_wrong_handle.mlir
module {
  func.func @wrong_samp_handle() attributes {gfx.stage = #gfx.stage<fragment>} {
    %samp = gfx.sampler_binding 0, 0, "samp" : !gfx.handle<texture_2d>
    %col = arith.constant dense<1.0> : vector<4xf32>
    gfx.output %col 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_sample_swapped_handles.mlir
module {
  func.func @swapped_handles() attributes {gfx.stage = #gfx.stage<fragment>} {
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %samp = gfx.sampler_binding 0, 1, "samp" : !gfx.handle<sampler>
    %col = gfx.texture_sample %samp, %tex, %uv : (!gfx.handle<sampler>, !gfx.handle<texture_2d>, vector<2xf32>) -> vector<4xf32>
    gfx.output %col 0 "color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_load_float_coords.mlir
module {
  func.func @bad_load_coords() attributes {gfx.stage = #gfx.stage<fragment>} {
    %tex = gfx.texture_binding 0, 0, "tex" : !gfx.handle<texture_2d>
    %uv = arith.constant dense<[0.5, 0.5]> : vector<2xf32>
    %lod = arith.constant 0 : i32
    %col = gfx.texture_load %tex, %uv, %lod : (!gfx.handle<texture_2d>, vector<2xf32>, i32) -> vector<4xf32>
    gfx.output %col 0 "color" <perspective> : vector<4xf32>
    return
  }
}
