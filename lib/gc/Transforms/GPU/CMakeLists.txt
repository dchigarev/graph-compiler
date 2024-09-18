gc_add_mlir_library(GcGpuPasses
  GpuToGpuOcl.cpp
  LinalgToXeGPU.cpp
  Pipeline.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRXeGPUDialect
    MLIRGPUTransforms
    MLIRGPUToSPIRV
    MLIRSCFToGPU
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMemRefTransforms
    GcInterface
    GcUtilsIR
)
  
include(imex)
get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
target_include_directories(GcGpuPasses PRIVATE ${IMEX_INCLUDES})

# include(imex)
# get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
# get_property(IMEX_DIALECT_LIBS GLOBAL PROPERTY IMEX_DIALECT_LIBS)
# get_property(IMEX_CONVERSION_LIBS GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
# set(IMEX_LIBS
#         ${IMEX_DIALECT_LIBS}
#         ${IMEX_CONVERSION_LIBS}
#         IMEXTransforms
#         IMEXUtil
# )

# foreach (include_dir IN LISTS IMEX_INCLUDES)
#   list(APPEND IMEX_BUILD_INCLUDES $<BUILD_INTERFACE:${include_dir}>)
# endforeach ()

# target_include_directories(GcGpuPasses PUBLIC ${IMEX_BUILD_INCLUDES})
# target_link_libraries(GcGpuPasses PUBLIC ${IMEX_LIBS})
# target_link_libraries(GcPasses PUBLIC GcGpuPasses)
# set_property(GLOBAL APPEND PROPERTY IMEX_LIBS ${IMEX_LIBS})
