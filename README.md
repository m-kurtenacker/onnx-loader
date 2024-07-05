# Requirements

All the requirements for AnyDSL are obviously still applicable.

Additionally, we require packages to build a python venv during the build phase.
On Ubuntu, this would be the python3-venv package.

At the time of writing, ONNX has some strange behavior if it is being built with gcc above version 12.
This relates to Protobuf and Absl, which in turn are being built by the ONNX project whenever they are not found elsewhere.
It is possible to avoid these issues by installing an external version of Protobuf and Absl.
However you attempt to build it, the ONNX libraries are required in the end, both as a library for the application to link to, as well as the python environment to interact with networks.
Do what you must.

# Supported Compile Flags

    - `TARGET_NETWORK`: Which network to actually compile this for.

## Finding or building the correct version of AnyDSL

    - `ONNX_PKG_AnyDSL_AUTOBUILD`: Builds AnyDSL in this repository, checking out the correct branches for plugin execution, building half, json, and RV automatically.
      All the cmake definitions for compiling AnyDSL can be set.
      However, you have to make sure that anyopt is being build, and that plugin execution is available.
      For most intents and purposes, the default flags should suffice.
    - `AnyDSL_GIT_URL`: Where the AnyDSL meta repository should be loaded from.

## Finding or building the correct version of LLVM

AnyDSL relies on LLVM, and RV.
There are a number of options to make this work:
Either, you bring your own build of LLVM and RV.
For this, set `LLVM_DIR` and `RV_DIR`.
Alternatively, you can set `AnyDSL_PKG_LLVM_AUTOBUILD` and `AnyDSL_PKG_RV_AUTOBUILD`, so both are being build from the AnyDSL project.
Building LLVM can be time consuming, but we rely on a specific version of LLVM, so this might be neccessary.
As a theird option, you can bring your own version of LLVM, and use `AnyDSL_PKG_RV_AUTOBUILD` to build RV through AnyDSL.
This is the default option right now, but it required you to bring your own build of LLVM.
On Ubuntu, using a prebuild version of LLVM from their website should be good enough.

P.S.: The dron CI system uses this last approach. .drone.d/download_llvm.sh will conventiently download and extract a prebuild version of llvm to be used in this project.

    - `AnyDSL_PKG_LLVM_AUTOBUILD`: Let AnyDSL build LLVM as a dependency.
    - `AnyDSL_PKG_RV_AUTOBUILD`: Let AnyDSL build RV as a dependency.
    - `LLVM_DIR`: Path to a preinstalled LLVM
    - `RV_DIR`: Path to a preinstalled RV

## Testing flags

    - `BUILD_TESTING`: As the name implies, builds a small set of tests that checks all operators perform as expected.
      The test suite contains both checks that execute the operators individually, as well as small integration tests that execute a small example network and cross-checks the results with onnx runtime as a refference implementation.
