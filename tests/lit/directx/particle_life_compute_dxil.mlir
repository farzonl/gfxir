// REQUIRES: llc, dxv
// RUN: %gfx-opt %s --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-MLIR-LLVM
// RUN: %gfx-compile %s --input-format mlir --emit-llvm -S | %FileCheck %s --check-prefix=CHECK-LLVM-IR
// RUN: %gfx-compile %s --input-format mlir --target directx --output %t.ll
// RUN: %llc -O0 -filetype=obj %t.ll -o %t.unsigned.dxil
// RUN: %dxv %t.unsigned.dxil -o %t.dxil

// CHECK-MLIR-LLVM-LABEL: llvm.func @particle_life_step()
// CHECK-MLIR-LLVM-SAME: "hlsl.shader", "compute"
// CHECK-MLIR-LLVM-SAME: "hlsl.numthreads", "64,1,1"
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.resource.handlefrombinding"
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.resource.getpointer"
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.thread.id"

// CHECK-LLVM-IR: target triple = "dxil-pc-shadermodel6.6-compute"
// CHECK-LLVM-IR: define void @particle_life_step()
// Compact SRVs t0..t3 (isWriteable=0) and UAVs u0..u1 (isWriteable=1) in space 0:
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 1, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 2, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(i32 0, i32 3, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 1, i32 1, i32 0, ptr null)
// CHECK-LLVM-IR-NOT: !dx.semantic.signatures
// CHECK-LLVM-IR: !dx.rootsignatures
// CHECK-LLVM-IR-DAG: !"RootConstants"
// CHECK-LLVM-IR-DAG: !"UAV", i32 2, i32 0, i32 0
// CHECK-LLVM-IR-DAG: !"SRV", i32 4, i32 0, i32 0
// CHECK-LLVM-IR-DAG: !"DescriptorTable"

module attributes {
  gfx.parameters = #gfx.parameter_block<"particle_life_params", [
    #gfx.parameter_field<"count", i32>,
    #gfx.parameter_field<"num_species", i32>,
    #gfx.parameter_field<"dt", f32>,
    #gfx.parameter_field<"friction", f32>,
    #gfx.parameter_field<"r_max", f32>,
    #gfx.parameter_field<"r_min", f32>,
    #gfx.parameter_field<"world_width", f32>,
    #gfx.parameter_field<"world_height", f32>,
    #gfx.parameter_field<"force_scale", f32>
  ]>
} {
  func.func @particle_life_step(
      %in_pos: !gfx.ptr<vector<2xf32>, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>},
      %in_vel: !gfx.ptr<vector<2xf32>, global> {gfx.binding = 1 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>},
      %in_species: !gfx.ptr<i32, global> {gfx.binding = 2 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>},
      %out_pos: !gfx.ptr<vector<2xf32>, global> {gfx.binding = 3 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>},
      %out_vel: !gfx.ptr<vector<2xf32>, global> {gfx.binding = 4 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>},
      %force_matrix: !gfx.ptr<f32, global> {gfx.binding = 5 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %i = vector.extract %gid[0] : i32 from vector<3xi32>

    %count = gfx.parameter "particle_life_params" "count" : i32
    %num_species = gfx.parameter "particle_life_params" "num_species" : i32
    %dt = gfx.parameter "particle_life_params" "dt" : f32
    %friction = gfx.parameter "particle_life_params" "friction" : f32
    %r_max = gfx.parameter "particle_life_params" "r_max" : f32
    %r_min = gfx.parameter "particle_life_params" "r_min" : f32
    %world_w = gfx.parameter "particle_life_params" "world_width" : f32
    %world_h = gfx.parameter "particle_life_params" "world_height" : f32
    %force_scale = gfx.parameter "particle_life_params" "force_scale" : f32

    %in_bounds = arith.cmpi ult, %i, %count : i32
    scf.if %in_bounds {
      %p_i_ptr = gfx.ptr_offset %in_pos, %i : (!gfx.ptr<vector<2xf32>, global>, i32) -> !gfx.ptr<vector<2xf32>, global>
      %p_i = gfx.load %p_i_ptr : (!gfx.ptr<vector<2xf32>, global>) -> vector<2xf32>
      %xi = vector.extract %p_i[0] : f32 from vector<2xf32>
      %yi = vector.extract %p_i[1] : f32 from vector<2xf32>

      %v_i_ptr = gfx.ptr_offset %in_vel, %i : (!gfx.ptr<vector<2xf32>, global>, i32) -> !gfx.ptr<vector<2xf32>, global>
      %v_i = gfx.load %v_i_ptr : (!gfx.ptr<vector<2xf32>, global>) -> vector<2xf32>

      %s_i_ptr = gfx.ptr_offset %in_species, %i : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
      %s_i = gfx.load %s_i_ptr : (!gfx.ptr<i32, global>) -> i32

      %zero_idx = arith.constant 0 : index
      %one_idx = arith.constant 1 : index
      %count_idx = arith.index_cast %count : i32 to index
      %zero_f32 = arith.constant 0.0 : f32
      %one_f32 = arith.constant 1.0 : f32
      %two_f32 = arith.constant 2.0 : f32
      %half_f32 = arith.constant 0.5 : f32
      %eps_f32 = arith.constant 0.0001 : f32

      %forces:2 = scf.for %j_idx = %zero_idx to %count_idx step %one_idx
          iter_args(%fx = %zero_f32, %fy = %zero_f32) -> (f32, f32) {
        %j = arith.index_cast %j_idx : index to i32
        %is_self = arith.cmpi eq, %i, %j : i32

        %next_f:2 = scf.if %is_self -> (f32, f32) {
          scf.yield %fx, %fy : f32, f32
        } else {
          %p_j_ptr = gfx.ptr_offset %in_pos, %j : (!gfx.ptr<vector<2xf32>, global>, i32) -> !gfx.ptr<vector<2xf32>, global>
          %p_j = gfx.load %p_j_ptr : (!gfx.ptr<vector<2xf32>, global>) -> vector<2xf32>
          %xj = vector.extract %p_j[0] : f32 from vector<2xf32>
          %yj = vector.extract %p_j[1] : f32 from vector<2xf32>

          %s_j_ptr = gfx.ptr_offset %in_species, %j : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
          %s_j = gfx.load %s_j_ptr : (!gfx.ptr<i32, global>) -> i32

          %raw_dx = arith.subf %xj, %xi : f32
          %raw_dy = arith.subf %yj, %yi : f32

          %half_w = arith.mulf %world_w, %half_f32 : f32
          %half_h = arith.mulf %world_h, %half_f32 : f32

          %dx_gt = arith.cmpf ogt, %raw_dx, %half_w : f32
          %dx_sub = arith.subf %raw_dx, %world_w : f32
          %neg_half_w = arith.subf %zero_f32, %half_w : f32
          %dx_lt_neg = arith.cmpf olt, %raw_dx, %neg_half_w : f32
          %dx_add = arith.addf %raw_dx, %world_w : f32
          %dx0 = arith.select %dx_gt, %dx_sub, %raw_dx : f32
          %dx = arith.select %dx_lt_neg, %dx_add, %dx0 : f32

          %neg_half_h = arith.subf %zero_f32, %half_h : f32
          %dy_gt = arith.cmpf ogt, %raw_dy, %half_h : f32
          %dy_sub = arith.subf %raw_dy, %world_h : f32
          %dy_lt_neg = arith.cmpf olt, %raw_dy, %neg_half_h : f32
          %dy_add = arith.addf %raw_dy, %world_h : f32
          %dy0 = arith.select %dy_gt, %dy_sub, %raw_dy : f32
          %dy = arith.select %dy_lt_neg, %dy_add, %dy0 : f32

          %dx2 = arith.mulf %dx, %dx : f32
          %dy2 = arith.mulf %dy, %dy : f32
          %r2 = arith.addf %dx2, %dy2 : f32
          %r = math.sqrt %r2 : f32

          %within_cutoff = arith.cmpf olt, %r, %r_max : f32
          %outside_singularity = arith.cmpf ogt, %r, %eps_f32 : f32
          %valid_pair = arith.andi %within_cutoff, %outside_singularity : i1

          %df:2 = scf.if %valid_pair -> (f32, f32) {
            %inv_r = math.rsqrt %r2 : f32
            %ux = arith.mulf %dx, %inv_r : f32
            %uy = arith.mulf %dy, %inv_r : f32

            %in_repulsion = arith.cmpf olt, %r, %r_min : f32
            %f_mag = scf.if %in_repulsion -> (f32) {
              %ratio = arith.divf %r, %r_min : f32
              %f_rep = arith.subf %ratio, %one_f32 : f32
              scf.yield %f_rep : f32
            } else {
              %num = arith.subf %r, %r_min : f32
              %den = arith.subf %r_max, %r_min : f32
              %xi_dist = arith.divf %num, %den : f32
              %two_xi = arith.mulf %two_f32, %xi_dist : f32
              %term = arith.subf %two_xi, %one_f32 : f32
              %abs_term = math.absf %term : f32
              %tent = arith.subf %one_f32, %abs_term : f32

              %mat_row = arith.muli %s_i, %num_species : i32
              %mat_idx = arith.addi %mat_row, %s_j : i32
              %coeff_ptr = gfx.ptr_offset %force_matrix, %mat_idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
              %coeff = gfx.load %coeff_ptr : (!gfx.ptr<f32, global>) -> f32
              %f_attr = arith.mulf %coeff, %tent : f32
              scf.yield %f_attr : f32
            }
            %f_scaled = arith.mulf %f_mag, %force_scale : f32
            %dfx = arith.mulf %ux, %f_scaled : f32
            %dfy = arith.mulf %uy, %f_scaled : f32
            scf.yield %dfx, %dfy : f32, f32
          } else {
            scf.yield %zero_f32, %zero_f32 : f32, f32
          }
          %acc_fx = arith.addf %fx, %df#0 : f32
          %acc_fy = arith.addf %fy, %df#1 : f32
          scf.yield %acc_fx, %acc_fy : f32, f32
        }
        scf.yield %next_f#0, %next_f#1 : f32, f32
      }

      %vxi = vector.extract %v_i[0] : f32 from vector<2xf32>
      %vyi = vector.extract %v_i[1] : f32 from vector<2xf32>

      %f_dt_x = arith.mulf %forces#0, %dt : f32
      %f_dt_y = arith.mulf %forces#1, %dt : f32
      %vx_acc = arith.addf %vxi, %f_dt_x : f32
      %vy_acc = arith.addf %vyi, %f_dt_y : f32

      %fric_dt = arith.mulf %friction, %dt : f32
      %damp = arith.subf %one_f32, %fric_dt : f32
      %damp_clamped = arith.maxnumf %damp, %zero_f32 : f32

      %vx_next = arith.mulf %vx_acc, %damp_clamped : f32
      %vy_next = arith.mulf %vy_acc, %damp_clamped : f32
      %v_next = vector.from_elements %vx_next, %vy_next : vector<2xf32>

      %dx_pos = arith.mulf %vx_next, %dt : f32
      %dy_pos = arith.mulf %vy_next, %dt : f32
      %x_raw = arith.addf %xi, %dx_pos : f32
      %y_raw = arith.addf %yi, %dy_pos : f32

      %x_div = arith.divf %x_raw, %world_w : f32
      %x_floor = math.floor %x_div : f32
      %x_mod_w = arith.mulf %x_floor, %world_w : f32
      %x_rem = arith.subf %x_raw, %x_mod_w : f32
      %x_pos = arith.maxnumf %x_rem, %zero_f32 : f32
      %x_ge_w = arith.cmpf oge, %x_pos, %world_w : f32
      %x_wrapped = arith.select %x_ge_w, %zero_f32, %x_pos : f32

      %y_div = arith.divf %y_raw, %world_h : f32
      %y_floor = math.floor %y_div : f32
      %y_mod_h = arith.mulf %y_floor, %world_h : f32
      %y_rem = arith.subf %y_raw, %y_mod_h : f32
      %y_pos = arith.maxnumf %y_rem, %zero_f32 : f32
      %y_ge_h = arith.cmpf oge, %y_pos, %world_h : f32
      %y_wrapped = arith.select %y_ge_h, %zero_f32, %y_pos : f32

      %p_next = vector.from_elements %x_wrapped, %y_wrapped : vector<2xf32>

      %out_p_ptr = gfx.ptr_offset %out_pos, %i : (!gfx.ptr<vector<2xf32>, global>, i32) -> !gfx.ptr<vector<2xf32>, global>
      gfx.store %p_next, %out_p_ptr : (vector<2xf32>, !gfx.ptr<vector<2xf32>, global>)

      %out_v_ptr = gfx.ptr_offset %out_vel, %i : (!gfx.ptr<vector<2xf32>, global>, i32) -> !gfx.ptr<vector<2xf32>, global>
      gfx.store %v_next, %out_v_ptr : (vector<2xf32>, !gfx.ptr<vector<2xf32>, global>)
    }
    return
  }
}
