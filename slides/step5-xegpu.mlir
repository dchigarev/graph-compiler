module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%c4, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}