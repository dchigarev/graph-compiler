// RUN: gc-opt %s --remove-allocs --split-input-file | FileCheck %s

// A dequantized weight tile is written as a rank-3 vector and read back as the
// rank-2 operand of a contraction: the leading unit dim is implicit in the
// transfers, with no reshape op between the alloc and the accesses.
func.func @forward_implicit_leading_unit_dim(%v: vector<1x64x16xf16>)
    -> vector<64x16xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1x64x16xf16>
  vector.transfer_write %v, %alloc[%c0, %c0, %c0]
      {in_bounds = [true, true, true]}
      : vector<1x64x16xf16>, memref<1x64x16xf16>
  %r = vector.transfer_read %alloc[%c0, %c0, %c0], %pad
      {in_bounds = [true, true]} : memref<1x64x16xf16>, vector<64x16xf16>
  return %r : vector<64x16xf16>
}

// CHECK-LABEL: func.func @forward_implicit_leading_unit_dim
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     vector.transfer
// CHECK:         %[[R:.+]] = vector.shape_cast %arg0 : vector<1x64x16xf16> to vector<64x16xf16>
// CHECK:         return %[[R]]

// -----

// The same buffer after the unit dims have been dropped from the transfers:
// both accesses reach the alloc through a rank-reducing subview.
func.func @forward_through_rank_reducing_subview(%v: vector<64x16xf16>)
    -> vector<64x16xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1x64x16xf16>
  %w = memref.subview %alloc[0, 0, 0] [1, 64, 16] [1, 1, 1]
      : memref<1x64x16xf16> to memref<64x16xf16>
  vector.transfer_write %v, %w[%c0, %c0] {in_bounds = [true, true]}
      : vector<64x16xf16>, memref<64x16xf16>
  %rv = memref.subview %alloc[0, 0, 0] [1, 64, 16] [1, 1, 1]
      : memref<1x64x16xf16> to memref<64x16xf16>
  %r = vector.transfer_read %rv[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<64x16xf16>, vector<64x16xf16>
  return %r : vector<64x16xf16>
}

// CHECK-LABEL: func.func @forward_through_rank_reducing_subview
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.subview
// CHECK-NOT:     vector.transfer
// CHECK:         return %arg0

// -----

// The buffer is allocated outside the k-loop and written/read inside it, so the
// write dominates the read from an enclosing region.
func.func @forward_inside_loop(%v: vector<1x64x16xf16>, %n: index)
    -> vector<64x16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.000000e+00 : f16
  %init = arith.constant dense<0.000000e+00> : vector<64x16xf16>
  %alloc = memref.alloc() : memref<1x64x16xf16>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init)
      -> (vector<64x16xf16>) {
    vector.transfer_write %v, %alloc[%c0, %c0, %c0]
        {in_bounds = [true, true, true]}
        : vector<1x64x16xf16>, memref<1x64x16xf16>
    %b = vector.transfer_read %alloc[%c0, %c0, %c0], %pad
        {in_bounds = [true, true]} : memref<1x64x16xf16>, vector<64x16xf16>
    %s = arith.addf %acc, %b : vector<64x16xf16>
    scf.yield %s : vector<64x16xf16>
  }
  memref.dealloc %alloc : memref<1x64x16xf16>
  return %r : vector<64x16xf16>
}

// CHECK-LABEL: func.func @forward_inside_loop
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK-NOT:       vector.transfer
// CHECK:           %[[B:.+]] = vector.shape_cast %arg0
// CHECK:           arith.addf %{{.+}}, %[[B]]
// CHECK-NOT:     memref.dealloc

// -----

// The write only covers the first half of the buffer, so the read observes
// uninitialized memory and the buffer must be kept.
func.func @keep_partially_written_alloc(%v: vector<32x16xf16>)
    -> vector<64x16xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1x64x16xf16>
  vector.transfer_write %v, %alloc[%c0, %c0, %c0] {in_bounds = [true, true]}
      : vector<32x16xf16>, memref<1x64x16xf16>
  %r = vector.transfer_read %alloc[%c0, %c0, %c0], %pad
      {in_bounds = [true, true]} : memref<1x64x16xf16>, vector<64x16xf16>
  return %r : vector<64x16xf16>
}

// CHECK-LABEL: func.func @keep_partially_written_alloc
// CHECK:         memref.alloc
// CHECK:         vector.transfer_write
// CHECK:         vector.transfer_read

// -----

// The read is not dominated by the write and must not be forwarded.
func.func @keep_alloc_read_before_write(%v: vector<1x64x16xf16>)
    -> vector<64x16xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1x64x16xf16>
  %r = vector.transfer_read %alloc[%c0, %c0, %c0], %pad
      {in_bounds = [true, true]} : memref<1x64x16xf16>, vector<64x16xf16>
  vector.transfer_write %v, %alloc[%c0, %c0, %c0]
      {in_bounds = [true, true, true]}
      : vector<1x64x16xf16>, memref<1x64x16xf16>
  return %r : vector<64x16xf16>
}

// CHECK-LABEL: func.func @keep_alloc_read_before_write
// CHECK:         memref.alloc
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_write
