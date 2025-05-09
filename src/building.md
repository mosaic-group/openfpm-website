# Building from source


## List of Supported OS

| OS | Architecture | Support |
|---|---|---|
| Linux | x86-64/arm64 | Yes/No |
| macOS | x86-64/arm64 | Yes/Yes |
| FreeBSD | x86-64/arm64 | No/No |
| Windows (*only with Cygwin) | x86-64/arm64 | Yes/No |

---

## Git Subprojects
[OpenFPM](https://github.com/mosaic-group/openfpm) project includes the following subprojects

| Submodule | Purpose | Depends on |
|---|---|---|
| [openfpm_devices](https://github.com/mosaic-group/openfpm_devices) | Memory management, GPU primitives | |
| [openfpm_data](https://github.com/mosaic-group/openfpm_data) | Serial data structures | openfpm_devices |
| [openfpm_vcluster](https://github.com/mosaic-group/openfpm_vcluster) | Parallel communication | openfpm_data |
| [openfpm_io](https://github.com/mosaic-group/openfpm_io) | Serial/Parallel Input-Output | openfpm_data, openfpm_vcluster |
| [openfpm_pdata](https://github.com/mosaic-group/openfpm_pdata) | Parallel data structures | openfpm_devices, openfpm_data, openfpm_vcluster, openfpm_io |
| [openfpm_numerics](https://github.com/mosaic-group/openfpm_numerics) | Numerical algorithms | openfpm_data, openfpm_vcluster, openfpm_io, openfpm_pdata |

---


The subprojects are managed using _Git submodules_. Please refer to this [manual](https://git-scm.com/book/en/v2/Git-Tools-Submodules) on how to use this tool:
```sh
git clone https://github.com/mosaic-group/openfpm
cd openfpm
git submodule init
git submodule update 
# optional: switch to a non-master branch, e.g. develop
git checkout develop
git submodule foreach "git checkout develop"
git submodule foreach "git pull origin develop"
```

## Dependencies

### Install prerequisites

```sh
# for linux-based systems
apt-get install build-essential make cmake make cmake git bzip2 libbz2-dev python-dev wget
# or 
apt-get install gcc g++ gfortran libtool libxml2-dev libxslt-dev make cmake git bzip2 libbz2-dev python-dev wget 
# for other systems
yum install g++ gcc-gfortran libtool make cmake git bzip2 bzip2-devel python-devel libxml2-devel libxslt-devel wget
brew install gcc libtool make cmake git python bzip2 wget 
```

OpenFPM is build upon the following open-source tools. Please intall these by building from source or with a package manager. 

### Building dependencies from source


| Tool | Submodule | Description | Optional | OpenFPM Version  |
|---|---|---|---|---|
| [Open MPI](https://www.open-mpi.org/) | openfpm_vcluster | The Open MPI Project is an open source [Message Passing Interface](https://www.mpi-forum.org/) implementation | NO | 4.1.6 (building from source on cygwin not supported, has to be preinstalled) |
| [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) | openfpm_pdata |  METIS is a set of serial programs for partitioning graphs and producing fill reducing orderings for sparse matrices | YES (or ParMETIS) | 5.1.0 |
| [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview) | openfpm_pdata, openfpm_numerics | Extends the functionality of METIS and includes routines that are especially suited for parallel AMR computations and large scale numerical simulations | YES (or METIS) | 4.0.3 |
| [BOOST](https://www.boost.org/) | openfpm_data, openfpm_vcluster, openfpm_io, openfpm_pdata, openfpm_numerics | Set of libraries that provides support for templated data structures, multithreading, unit testing etc. | NO | 1.84.0 (*works on arm64 macOS only with `clang`) |
| [zlib](https://www.zlib.net/) | openfpm_io | Lossless data-compression library needed by HDF5 to deflate stored files | NO | 1.3.1 (*doesn't work on Windows) |
| [HDF5](https://www.hdfgroup.org/) | openfpm_io | Distributed file format that supports large, complex, heterogeneous data. Requires [zlib](https://www.zlib.net/) | NO | 1.14.3 |
| [Vc](https://web-docs.gsi.de/~mkretz/Vc-0.7/index.html) | openfpm_data | The library is a collection of SIMD vector classes with existing implementations for SSE, AVX, and a scalar fallback | NO | 1.4.4 |
| [libhilbert](https://www.hdfgroup.org/) | openfpm_data | Library producing Hilbert indices for multidimensional data to iterate through the grid elements following an Hilbert space filling curve. | NO | master (*no active support) |
| [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) | openfpm_devices | A C++ runtime API and kernel language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code. One of the alternative execution backends for CUDA-like code supported by OpenFPM | Yes |  |
| [alpaka](https://alpaka.readthedocs.io/en/latest/index.html) | openfpm_devices | A header-only C++17 abstraction library for accelerator development. One of the alternative execution backends for CUDA-like code supported by OpenFPM | Yes |  |
| [OpenBLAS](http://www.openblas.net/) | openfpm_numerics | An optimized BLAS (Basic Linear Algebra Subprograms) library, used for performing basic vector and matrix operations | NO* | 0.3.26 |
| [suitesparse](https://people.engr.tamu.edu/davis/suitesparse.html) | openfpm_numerics | A suite of sparse matrix algorithms. Here UMFPACK - multifrontal LU factorization module. Requires [OpenBLAS](http://www.openblas.net/) | NO* | 5.7.2 |
| [Eigen](https://eigen.tuxfamily.org/index.php) | openfpm_numerics | Template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Requires [suitesparse](https://people.engr.tamu.edu/davis/suitesparse.html) | Yes* (or Petsc) | 3.4.0 |
| [Blitz++](https://github.com/blitzpp/blitz) | openfpm_numerics | A meta-template library for array manipulation in C++ with a speed comparable to Fortran implementations, while preserving an object-oriented interface | NO* | 1.0.2 |
| [Algoim](https://algoim.github.io) | openfpm_numerics | A collection of high-order accurate numerical methods and C++ algorithms for working with implicitly-defined geometry and level set methods. Requires [Blitz++](https://github.com/blitzpp/blitz) | NO* | master |
| [PETSc](https://petsc.org/) | openfpm_numerics |  Scientific computation toolkit for linear and non-linear solvers, preconditioners, time integrators. Installs HYPRE, MUMPS, ScaLAPACK, SuperLU_DIST. Requires [OpenBLAS](http://www.openblas.net/), [suitesparse](https://people.engr.tamu.edu/davis/suitesparse.html), [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview) | Yes* (or Eigen) | 3.20.5 |
| [minter](https://git.mpi-cbg.de/mosaic/software/math/minter) | openfpm_numerics |  Multivariate interpolation module | No | header_only |

---

_*optional if openfpm_numerics is disabled_

OpenFPM uses [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) to move data from one GPU to another GPU (intranode and extranode) without moving the data to host. This feature requires that OpenMPI is compiled with CUDA support. It can without this feature as well using GPUDirect of an old version (1.0). In practice it requires that MPI works with host pinned memory allocated with CUDA. This feature has been introduced with CUDA 5.0 in 2010. At best of our knowledge this feature should work without special compilation options for OpenMPI. On the other end we found several problems with GPUDirect v1.0 and Infiniband cards, when OpenMPI is not compiled with CUDA support. If you are on a super-computer or a machine you did not set-up, we suggest to re-install OpenMPI with CUDA support using the options suggested. Alternatively you can use the OpenMPI already provied.

**If any of the OpenFPM dependencies is available system-wide the building from source step can be skipped!**

The following script installs OpenFPM dependencies to the directory `/home/test/openfpm_dependencies`, compiled libraries and headers of OpenFPM to `/home/test/openfpm_install`, uses _gcc_ toolkit, _4_ cores and no gpu support for OpenMPI.
```sh
# here for gcc/g++. Also for icc/icpc and clang/clang++
# to enable gpu accelerated code in MPI installed, set GPU_CUDA_SUPPORT=1
export CC=gcc \
  CXX=g++ \
  F77=gfortran \
  FC=gfortran \
  PREFIX_DEPENDS=/home/test/openfpm_dependencies \
  PREFIX_OPENFPM=/home/test/openfpm_install \
  NCORE=4 \
  GPU_CUDA_SUPPORT=0

./script/install_MPI.sh $PREFIX_DEPENDS $NCORE $GPU_CUDA_SUPPORT $CC $CXX $F77 $FC "--with-mpivendor=openmpi"
export PATH="$PREFIX_DEPENDS/MPI/bin:$PATH"

./script/install_Metis.sh $PREFIX_DEPENDS $NCORE $CC $CXX 
# Parmetis uses mpicc in make config by default
./script/install_Parmetis.sh $PREFIX_DEPENDS $NCORE 
./script/install_BOOST.sh $PREFIX_DEPENDS $NCORE
# Zlib uses CC=mpicc in ./configure
./script/install_ZLIB.sh $PREFIX_DEPENDS $NCORE
./script/install_HDF5.sh $PREFIX_DEPENDS $NCORE
./script/install_LIBHILBERT.sh $PREFIX_DEPENDS $NCORE
./script/install_VCDEVEL.sh $PREFIX_DEPENDS $NCORE $CC $CXX
# Install dependencies for numerics (optional)
./script/install_OPENBLAS.sh $PREFIX_DEPENDS $NCORE
./script/install_SUITESPARSE.sh $PREFIX_DEPENDS $NCORE
./script/install_EIGEN.sh $PREFIX_DEPENDS $NCORE
./script/install_BLITZ.sh $PREFIX_DEPENDS $NCORE
./script/install_ALGOIM.sh $PREFIX_DEPENDS $NCORE
./script/install_PETSC.sh $PREFIX_DEPENDS $NCORE $CC $CXX $F77 $FC
./script/install_MINTER.sh $PREFIX_DEPENDS

```

If the dependencies are not installed system-wide, but build from source, two environment variables have to be set accordingly:

- _LD_LIBRARY_PATH_ so the dynamic link loader knows where to search for the dynamic shared libraries;
- _PATH_ so the binary files could be executed without specifying the full path, e.q. `mpic++`

This could be done manually (e.g. by modifying _~/.bashrc_,_~/.zshrc_...) or with the following tool that produces the file
_openfpm_vars_. This file has to be sourced every time in a new session before running OpenFPM related code.

```sh
./script/create_env_vars.sh $PREFIX_DEPENDS $PREFIX_OPENFPM
source openfpm_vars
```

## Building OpenFPM

OpenFPM uses CMake build system. For it to function properly, CMake has to be able to locate
the dependencies of OpenFPM. If they are not installed system-wide, the following script passes their locations to CMake. Additionaly, the script assumes some _openfpm_ build parameters by default, e.g. build type, backend for CUDA-like code, enable/disable _numerics_ module, debug utilities etc. These parameters can be modified inside _script/conf_CMake.sh_
```sh
./script/conf_CMake.sh $PREFIX_DEPENDS $PREFIX_OPENFPM
```
The resultant CMake command is echoed to the terminal window and saved into the file _cmake_build_options_

```sh
mkdir build
cd build 
#insert the output of conf_CMake.sh and run the command
# <INSERT HERE>
make -j $NCORE
make install
cd ..
```
CMake config options (dependencies found, debug/error handling macros etc.) are exported as `#define OPTION` to `${CMAKE_BINARY_DIR}/config/config.h` header file to be included by the source files. The config file is generated per build and is unique for every directory where CMake command was run.

## Updating OpenFPM

To pull, build and install an up-to-date version of OpenFPM tracked by the local _Git_ branch (here _master_ or _develop_) run:

```sh
git pull origin master/develop
git submodule update
cd build
make -j $NCORE
make install
cd ..
```

## Running Tests and Examples

Optionally, all tests could be run in each module to assure the project and dependencies work correctly
```sh
cd openfpm_data
mpirun -np 1 ../build/openfpm_data/src/mem_map --run_test=\*/\* --log_level=test_suite
cd ..

cd openfpm_devices
mpirun -np 1 ../build/openfpm_devices/src/mem --run_test=\*/\* --log_level=test_suite
cd ..

cd openfpm_io
mpirun -np 1 ../build/openfpm_io/src/io --run_test=\*/\* --log_level=test_suite
cd ..

cd openfpm_numerics
mpirun -np 3 ../build/openfpm_numerics/src/numerics --run_test=\*/\* --log_level=test_suite
cd ..

cd openfpm_vcluster
mpirun -np 3 ../build/openfpm_vcluster/src/vcluster_test --run_test=\*/\* --log_level=test_suite
cd ..

cd openfpm_pdata
mpirun -np 3 ../build/openfpm_pdata/src/pdata --run_test=\*/\* --log_level=test_suite
cd ..
```

Example simulation codes are compiled with Makefile. For the dependencies (that are not installed system-wide) to be linked properly,
the following compiler options have to be set:

- _-llib_ Search for the library named _lib_ when linking
- _-Lloc_ The location _loc_ where to search for _lib_
- _-Idir_ Add the directory _dir_ to the list of directories to be searched for header files during preprocessing.

This could be done manually when compiling the example codes or with the following tool that produces the file _example.mk_.
_**Warning**: OpenFPM installed and the gpu examples have to be compiled in the same mode (i.e. via nvcc, hip, alpaka or gpu-emulated (SEQUENTIAL, OpenMP))_<br>
The file has to be placed in the folder _example_.
```sh
./script/create_example.mk.sh $PREFIX_DEPENDS $PREFIX_OPENFPM
mv example.mk example
```
The example codes are split in subfolders. To compile and run an example (e.g. _Vector/0_simple_) use _Makefile_
```sh
cd example/Vector/0_simple
make run
```
If dependencies or the codebase of OpenFPM change, to recompile an example use
```sh
make clean
make run
```

## Troubleshooting: known issues
- **Mac OS**: _building for macOS-x86_64 but attempting to link with file built for macOS-x86_64_. If the error is reporting the same architecure (might be _arm64_ as well), check whether _gcc_ and _runlib_ are showing the same backend consistantly, i.e. both have to be either _GNU_ or _clang_. If not, modify your _$PATH_ accordingly, e.g. in _.bashrc_ or _.zshrc_, and rebuild the project/dependency
- **Mac OS**: _CMake error: PETSC could not be found (missing: PETSC_EXECUTABLE_RUNS) (found version petsc...)_. Due to System Integrity Protection enabled, CMake module of PETSC might encounter problems when running a test program. This could be diagnosed by running `make check` in PETSC directory. The command will fail with `dyld[...]: Library not loaded`, while manually compiling the code samples (i.e. snes/tutorials) works. If the installation of PETSC works properly, the check PETSC_EXECUTABLE_RUNS in CMake could be disabled via `-DPETSC_EXECUTABLE_RUNS=True` added to the output of the command:
```sh
./script/conf_CMake.sh $PREFIX_DEPENDS $PREFIX_OPENFPM
```
- **Mac OS Xcode 15**: _Undefined symbols:  Linker command failed with exit code 1 (use -v to see invocation)_. In the new version Xcode 15 has introduced the new linker which might cause linking errors for LLVM back-end compilers (clang or gcc supplied by Apple). Try reverting to an old linker with:
```sh
# Make sure you export the environment variable before running CMake,
# as it is cached by CMake and becomes read-only when running make
export LDFLAGS="$LDFLAGS -Wl,-ld_classic"
cmake ..
```

---

In addition to the building from source described below, OpenFPM packages are
also available as pre-built [binaries](download.md) and [Docker images](docker.md)
