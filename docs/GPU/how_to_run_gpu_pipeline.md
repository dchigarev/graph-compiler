0. Clone graph-compiler and IMEX repositories:
  ```
  git clone https://github.com/intel/graph-compiler.git
  # need it to extract LLVM patches
  git clone https://github.com/intel/mlir-extensions.git
  ```
1. Clone LLVM and checkout to a pinned commit:
  ```
  export LLVM_COMMIT_HASH=$(< cmake/llvm-version.txt)
  git clone https://github.com/llvm/llvm-project
  cd llvm-project
  git checkout $LLVM_COMMIT_HASH
  ```
2. Apply LLVM patches from IMEX (we want to get rid of this step in the future)
  ```
  cd llvm-project
  git apply ../mlir-extensions/build_tools/patches/*
  ```
3. Build LLVM with `LLVM_BUILD_LLVM_DYLIB` and `LLVM_LINK_LLVM_DYLIB` **disabled:**
  ```
  cmake -G Ninja llvm -B build -DCMAKE_INSTALL_PREFIX=llvm-install \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_INSTALL_UTILS=true -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_INSTALL_GTEST=ON -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF![image](https://github.com/user-attachments/assets/784153e3-a5b8-4a82-b4aa-bfa593d31a22)
  cmake --build build --target install
  ```
4. Build graph-compiler with `-DGC_USE_GPU=ON`, `-DGC_TEST_ENABLE=OFF` (it won't build with tests enabled for some reason) and `-DIMEX_ENABLE_L0_RUNTIME=1` (it should propagate automatically with `-DGC_USE_GPU=ON` but it don't for some reason)
  ```
  cd graph-compiler
  mkdir build
  cd build
  cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DMLIR_DIR=../../llvm-project/llvm-install/lib/cmake/mlir \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DGC_USE_GPU=ON \
      -DGC_TEST_ENABLE=OFF \
      -DIMEX_ENABLE_L0_RUNTIME=1
  cmake --build .
  ```
5. 
