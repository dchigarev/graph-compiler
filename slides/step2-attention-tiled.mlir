// Shows an attention operation after it was tiled.
// We're tiling 'batch' and 'm' dimensions with tile sizes of 1 and 128 respectively.
// The tiled attention op is then decomposed into an flash-attention implementation.

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>

func.func @attention_f16(
    %arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>,
    %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>
) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %Q_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %K_slice = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %V_slice = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %O_slice = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
    %1 = linalgx.attention {
        gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8,
        indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO]
        }
        ins(
            %Q_slice, %K_slice, %V_slice, %cst 
            : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16
        ) 
        outs(%O_slice : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %0 : tensor<4x4096x64xf32>
}