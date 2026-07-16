gc_add_mlir_library(
  GcGpuOclPasses
  GpuToGpuOcl.cpp
  DEPENDS
  GraphCompilerPassIncGen
  LINK_LIBS
  PUBLIC
  GcInterface)
