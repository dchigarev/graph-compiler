func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  // wg-loop
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %1 = tensor.empty() : tensor<128xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128xf32>) -> tensor<128xf32>
    %3 = tensor.empty() : tensor<128xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
    %5 = tensor.empty() : tensor<128x64xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
    // k-loop computing running max and sum for softmax
    %7:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %6, %arg9 = %2, %arg10 = %4) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>) {
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_1[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %11 = tensor.empty() : tensor<64x64xf16>
      %transposed = linalg.transpose ins(%collapsed_4 : tensor<64x64xf16>) outs(%11 : tensor<64x64xf16>) permutation = [1, 0] 
      %12 = tensor.empty() : tensor<128x64xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %14 = linalg.matmul ins(%collapsed, %transposed : tensor<128x64xf16>, tensor<64x64xf16>) outs(%13 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %15 = tensor.empty() : tensor<128xf32>
      %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<128xf32>) -> tensor<128xf32>
      %reduced = linalg.reduce ins(%14 : tensor<128x64xf32>) outs(%16 : tensor<128xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %40 = arith.maximumf %in, %init : f32
          linalg.yield %40 : f32
        }
      %17 = tensor.empty() : tensor<128xf32>
      %18 = linalg.max ins(%arg9, %reduced : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
      %19 = tensor.empty() : tensor<128x64xf32>
      %broadcasted_5 = linalg.broadcast ins(%18 : tensor<128xf32>) outs(%19 : tensor<128x64xf32>) dimensions = [1] 
      %20 = tensor.empty() : tensor<128x64xf32>
      %21 = linalg.sub ins(%14, %broadcasted_5 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%20 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %22 = tensor.empty() : tensor<128x64xf32>
      %23 = linalg.exp ins(%21 : tensor<128x64xf32>) outs(%22 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %24 = tensor.empty() : tensor<128xf32>
      %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<128xf32>) -> tensor<128xf32>
      %reduced_6 = linalg.reduce ins(%23 : tensor<128x64xf32>) outs(%25 : tensor<128xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %40 = arith.addf %in, %init : f32
          linalg.yield %40 : f32
        }
      %26 = tensor.empty() : tensor<128xf32>
      %27 = linalg.sub ins(%arg9, %18 : tensor<128xf32>, tensor<128xf32>) outs(%26 : tensor<128xf32>) -> tensor<128xf32>
      %28 = tensor.empty() : tensor<128xf32>
      %29 = linalg.exp ins(%27 : tensor<128xf32>) outs(%28 : tensor<128xf32>) -> tensor<128xf32>
      %30 = tensor.empty() : tensor<128xf32>
      %31 = linalg.mul ins(%arg10, %29 : tensor<128xf32>, tensor<128xf32>) outs(%30 : tensor<128xf32>) -> tensor<128xf32>
      %32 = tensor.empty() : tensor<128xf32>
      %33 = linalg.add ins(%31, %reduced_6 : tensor<128xf32>, tensor<128xf32>) outs(%32 : tensor<128xf32>) -> tensor<128xf32>
      %34 = tensor.empty() : tensor<128x64xf32>
      %broadcasted_7 = linalg.broadcast ins(%29 : tensor<128xf32>) outs(%34 : tensor<128x64xf32>) dimensions = [1] 
      %35 = tensor.empty() : tensor<128x64xf32>
      %36 = linalg.mul ins(%arg8, %broadcasted_7 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%35 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %37 = tensor.empty() : tensor<128x64xf16>
      %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<128x64xf32>) outs(%37 : tensor<128x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %40 = arith.truncf %in : f32 to f16
        linalg.yield %40 : f16
      } -> tensor<128x64xf16>
      %extracted_slice_8 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_9 = tensor.collapse_shape %extracted_slice_8 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %39 = linalg.matmul ins(%38, %collapsed_9 : tensor<128x64xf16>, tensor<64x64xf16>) outs(%36 : tensor<128x64xf32>) -> tensor<128x64xf32>
      scf.yield %39, %18, %33 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
    }
    %8 = tensor.empty() : tensor<128x64xf32>
    %broadcasted = linalg.broadcast ins(%7#2 : tensor<128xf32>) outs(%8 : tensor<128x64xf32>) dimensions = [1] 
    %9 = tensor.empty() : tensor<128x64xf32>
    %10 = linalg.div ins(%7#0, %broadcasted : tensor<128x64xf32>, tensor<128x64xf32>) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %expanded = tensor.expand_shape %10 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %0 : tensor<4x4096x64xf32>
}