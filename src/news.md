## OpenFPM Versions

<img width=130px, style="margin:-20px"> | |
------------ | -----------------------------------------------------------------
**12/2/2024** | *OpenFPM 5.0.0*
**29/3/2022** | *OpenFPM 4.1.0*
**1/9/2021** | *OpenFPM 4.0.0*
**24/4/2021** | *OpenFPM 3.3.0*
**8/4/2021** | *OpenFPM 3.2.0*
**19/10/2020** | *OpenFPM 3.1.0*
**15/7/2020** | *OpenFPM 3.0.0*
**19/2/2019** | *OpenFPM 2.0.0*
**20/4/2018** | *OpenFPM 1.1.0*
**27/3/2018** | *OpenFPM 1.0.1*
**13/9/2017** | *OpenFPM 1.0.0*
**28/2/2017** | *OpenFPM 0.8.0*
**27/01/2017** | *OpenFPM 0.7.1*
**15/12/2016** | *OpenFPM 0.7.0*
**5/11/2016** | *OpenFPM 0.6.0*

# Change Log
## OpenFPM 5.0.0 - Feb 2024
- Move to `openfpm` meta OpenFPM project structure with git subprojects `openfpm_data`, `openfpm_devices`, `openfpm_io`, `openfpm_vcluster`, `openfpm_pdata`, `openfpm_numerics`. Example codes, installation scripts for dependencies, configuration scripts moved to `openfpm`. Only source code of `openfpm_pdata` kept in `openfpm_pdata`

### Added 
- Add DC-PSE GPU implementation
- Add support for _minter_ header-only library. Installed automatically if `openfpm_numerics` is enabled
- Add _Odeint_ GPU support

### Fixed
- Fix `remove_marked` when number of particles is decreased and memory is reused
- Fix `host_to_device_impl` when _allocated_ size > _actual_ size 
- Fix DC-PSE operator with Lagrange multiplier for number of variables > 1
- Fix template substitution failure for `Graph_CSR` with no edge
- Fix Cell List `getNNIterator` giving different neighborhood lists for dense and sparse versions
- Fix `sendrecvMultipleMessagesNBX` for message size > 2 GB

### Changes 
- Get rid of _autotools_ emulation for installation procedure. Remove `install/update/upgrade` scripts. Move to _CMake_ with manually building dependencies and manualy calling configuration scripts for _CMake_, `example.mk`, `openfpm_vars`
- Remove `gdbui` from list of git subprojects
- Move from _moderngpu_ as it's no longer supported to _cub_
- Refactor Cell List implementation, change code structure and variable naming. Split dense and sparse Cell List into partial template specializations
- Move all backends for CUDA emulation, CUDA headers and parallel primitives to `openfpm_devices` from `openfpm_data` 
- Remove default parameter from `NO_CHECK` from Cell List iterators (e.g. `getNNIterator`). `cl.template getNNIterator<NO_CHECK>(...)` is replaced by `cl.getNNIterator(...)`
- Pass Cell List `NNIteratorBox` parameter to `getCellList/getCellListGPU` of `vector_dist`. `getNNIteratorBox` has to be set when initializing the Cell List compared to `getNNIteratorRadius`, which could be set later, due to sparse Cell List on GPU using it in `construct()`. If not set, the dafault value of 1 would have been used by `construct()`

## OpenFPM 4.1.0 - Mar 2022
- On a general base the code should not use `CUDA_ON_CPU` but if it does `CUDA_ON_CPU` macro now cover both `SEQUENTIAL` and `OpenMP` backend. The macros `CUDIFY_USE_CUDA,CUDIFY_USE_HIP,CUDIFY_USE_OPENMP,CUDIFY_USE_SEQUENTIAL,CUDIFY_USE_NONE` can be checked to control which CUDA backend is used

### Fixed
- Minors bug

## OpenFPM 4.0.0 - Sep 2021

- Adding _DCPSE_, _Level-set_ based numerics (Closest-point)

### Changes

- Particles writer use now the new XML pvtp Paraview format
- Particles constructor does not accept discordance precision on space example:
            If I create a Particle set with `vector_dist<3,double,.......>`
            I MUST give in the constructor a domain `Box<3,double>` and a `Ghost<3,double>`, giving a `Ghost<3,float>` will produce
            an error

## OpenFPM 3.3.0 - Apr 2021

- Adding support for _HIP_ and _AMD_ GPU. (Only particles) [SPH example](vector-example.md#ex3) are compatible with HIP
Additional Notes:
- WARNING: _AMD_ GPUs are tested manually and not in CI. This mean that out this release stuff can break at least
           until I do not convince my working place to buy one for me ... and is gonna be hard because or rule here and there ... or who 
           is reading this message does not want to buy one for me :-)
- SparseGridGPU are unsupported untill AMD does not fix the bug reported [here](https://github.com/ROCm-Developer-Tools/HIP/issues/2260):

### Changes

- None

### Fixed

- uninitialized variables in the SPH example on GPU, and other fixes necessary for _AMD_ gpus

## OpenFPM 3.2.0 - Jan 2021

- Adding `CUDA_ON_CPU` option to run CUDA code on CPU
- Adding `gdb-gui` debugger

### Fixed

- Minors bugs

### Changes

- In order to compile OpenFPM is now required a compiler implementing C++14 Standard

## OpenFPM 3.1.0 - Oct 2020 

- Adding GPU support for `ghost_put`
- Adding support for CUDA 11

### Fixed

- Few details in the installation system of linear algebra for OSX

### Changed

- None

## OpenFPM 3.0.0 - Jul 2020

- Upgrading all the dependencies: `BOOST,PETSC,SUITESPARSE,OPENBLAS`
- Adding CPU and GPU sparse grids. Look at the examples SparseGrid in the forlder examples
- Improving performance of GPU function remove_marked

### Fixed

- Several installation bugs on PETSC installation

### Changed

- Name for GoogleCharts has been changed please look at the examples

## OpenFPM 1.X - End of life (Theese versions are not supported enymore )

## OpenFPM 2.0.0 - Feb 2019

### Added

- Adding GPU support (see example 1_gpu_first_step
				  3_molecular_dynamic_gpu 
                                  7_sph_dlb_gpu 
                                  7_sph_dlb_gpu_opt)

### Fixed 

- Detection of clang 10.0.0 on mac-osx mojave
- In VTK binary format all 64 bit types are casted to 32 bit. Either the long/unsigned_long are bugged in Paraview we tested, either I do not understand how they work.

### Changed

- The type Vcluster now is templated and the standard Vcluster is Vcluster<>
           Most probably you have to change in your code from Vcluster to Vcluster<>

## OpenFPM 1.1.1 - Dec 2018

### Fixed 

- Detection of clang 10.0.0 on mac-osx mojave

## OpenFPM 1.0.X - End of life (Theese versions are not supported enymore)

## OpenFPM 1.1.0 - Feb 2018

### Added

- Interface for Multi-vector dynamic load balancing
- Increaded performance for grid ghost get
- Introduced forms to increase the performance of the grid iterator in case of stencil code (see example 5_GrayScott)
- EMatrix wrapped eigen matrices compatibles with vector_dist_id
- General tuning for high dimension vector_dist_id (up to 50 dimensions) + PS_CMA_ES (Particle-Swarm Covariant Matrix Adaptation Evolution Strategy) example in Numerics
- Added Discrete element Method example (8_DEM)
- Added serial_to_parallel example VCluster (2_serial_to_parallel). The example it show how to port a serial example into openfpm gradually swtiching from
  a serial section to a parallel section
- Introduced map(LOCAL) for fast communication in case we have small movement

### Fixed

- Installation/detection of PETSC
- CRITICAL-BUG scalar product in combination with vector product is broken (it return 0)
- Fixing 2D IO in binary for vector
- Fixing 1D grid writer in ASCII mode
- Fixing Intel compilation of Linear algebra

## OpenFPM 1.0.0 - Sep 2017

### Added
- Introduced getDomainIterator for Cell-list
- New dynamic load balancing scheme to 7_SPH_opt (see Vector/7_SPH_opt)
- Increased performance of 7_SPH_opt
- Vortex in Cell example Numerics/Vortex_in_cell
- Interpolation functions (see Numerics/vortex_in_cell example)
- Gray-scott 3D example with stencil iterator optimization (see Grid/gray_scott_3d example)
- HDF5 Check point restart for vector_dist particles (see Vector/1_HDF5_save_and_load) 
- Raw reader for grid (see ...)
- Added setPropNames to give names to properties (Grid and Vector see Vector/0_simple)
- Ghost put on grid (see Numerics/Vortex_in_Cell example)
- Stencil iterators for faster stencil codes see (Test3D_stencil function in Units tests src/Grid/Iterators/grid_dist_id_iterators_unit_tests.hpp)
- Algebraic multigrid solvers interface for linear systems (see Vortex in Cell example)
- Added setPropNames in vector_dist see Vector/0_simple
- Support for Windows with CYGWIN

### Fixed
- Bug fixes in installation of PETSC
- 2 Bugs in 7_SPH_opt and 7_SPH_opt error in Kernel and update for boundary particles
- Bug in VTK writer binary in case of vectors
- Bug in VTK writer binary: long int are not supported removing output
- Bug in FDScheme in the constructor with stencil bigger than one
- Bug Fixed Memory leak in petsc solver
- Bug Performance bug in the grid iterator

### Changed
- CAREFULL: write("output",frame) now has changed to write_frame("output",frame)
            write() with two arguments has a different meanings write("output",options)
-  getCellList and getCellListSym now return respectively
	  CellList_gen<dim, St, Process_keys_lin, Mem_fast, shift<dim, St>>
          CellList<dim, St, Mem_fast, shift<dim, St>>
    
- getIterator in CellList changed getCellIterator
- Grid iterator types has changes (one additional template parameter)
- FDScheme the constructor now has one parameter less (Parameter number 4 has been removed) (see Stokes_Flow examples in Numerics)
- MPI,PETSC,SUPERLU,Trilinos,MUMPS,SuiteSparse has been upgraded

## OpenFPM 0.8.0 - Feb 2017

### Added
- Dynamic Load balancing
- Added SPH Dam break with Dynamic load balancing (7_sph_dlb)(7_sph_dlb_opt)
- Added automatic procedure for update ./install --update and --upgrade
  (From 0.8.0 version will have a long term support for bug fixing. End-of-life of 0.8.0 is not decided yet, it should be still supported for bug fixing after release 0.9.0)
- Added video lessons for Dynamic load balancing (openfpm.mpi-cbg.de)
  (website officially open)
- Added for debugging the options PRINT_STACKTRACE, CHECKFOR_POSNAN, CHECKFOR_POSINF, CHECKFOR_PROPINF, CHECKFOR_PROPNAN, SE_CLASS3.
- Added the possibility to write binary VTK files using VTK_WRITER and FORMAT_BINARY see 0_simple_vector for an example

### Fixed
- Installation of PETSC with MUMPS  

### Changed
- BOOST updated to 1.63
- Eigen updated to 3.3.7

## OpenFPM 0.7.1 - Jan 2017

### Fixed
- Multiphase verlet single to all case generate overflow

## OpenFPM 0.7.0 - Dec 2016

### Added
- Symmetric cell-list/verlet list Crossing scheme
- VCluster examples
- Cell-list crossing scheme

### Fixed
- CRITICAL BUG: OpenFPM has a bug handling decomposition when a processor has a disconnected domains
                (By experience this case has been seen on big number of processors).
- Found and fixed a memory leak when using complex properties

### Changed
- The file VCluster has been mooved #include "VCluster.hpp" must be changed to #include "VCluster/VCluster.hpp"
  BECAUSE OF THIS, PLEASE CLEAN THE OPENFPM FOLDER OTHERWISE YOU WILL END TO HAVE 2 VCLUSTER.HPP

## OpenFPM 0.6.0 - Nov 2016

### Added
- Symmetric cell-list/verlet list
- Multi-phase cell-list and Multi-phase cell-list
- Added ghost_get that keep properties
- Examples: 1_ghost_get_put it show how to use ghost_get and put with the new options
            4_multiphase_celllist_verlet completely rewritten for new Cell-list and multiphase verlet
	    5_molecular_dynamic use case of symmetric cell-list and verlet list with ghost put
	    6_complex_usage It show how the flexibility of openfpm can be used to debug your program
- Plotting system can export graph in svg (to be included in the paper)

 
### Fixed
- Option NO_POSITION was untested
- Regression: Examples code compilation was broken on OSX (Affect only 0.5.1)
              (Internal: Added OSX examples compilarion/running test in the release pipeline)
- gray_scott example code (variable not initialized)


### Changes


## OpenFPM 0.5.1 - Sep 2016

### Added
- ghost_put support for particles
- Full-Support for complex property on vector_dist (Serialization)
- Added examples for serialization of complex properties 4_Vector
- improved speed of the iterators

### Fixed
- Installation PETSC installation fail in case of preinstalled MPI
- Miss-compilation of SUITESPARSE on gcc-6.2
- vector_dist with negative domain (Now supported)
- Grid 1D has been fixed
- One constructor of Box had arguments inverted.
  PLEASE CAREFULL ON THIS BUG
     float xmin[] = {0.0,0.0};
     float xmax[] = {1.0,1.0};
     // Box<2,float> box(xmax,xmin)    BUG IT WAS xmax,xmin
	 Box<2,float> box(xmin,xmax)  <--- NOW IT IS xmin,xmax
	 Box<2,float> box({0.0,0.0},{1.0,1.0}) <---- This constructor is not affected by the BUG

### Changed
- On gcc the -fext-numeric-literals compilation flag is now mandatory

## OpenFPM 0.5.0 - Aug 2016

### Added
- map communicate particles across processors mooving the information of all the particle map_list give the possibility to give a list of property to move from one to another processor
- Numeric: Finite Differences discretization with matrix contruction and parallel solvers (See example ... )
- vector_dist now support complex object like Point VectorS Box ... , with no limitation
   and more generic object like std::vector ... (WARNING TEMPORARY LIMITATION: Communication is not supported property must be excluded from communication using map_list and ghost_get)
- vector_dist support expressions (See example ...)
- No limit to ghost extension (they can be arbitrary extended)
- Multi-phase CellList
- Hilber curve data and computation reordering for cache firndliness

### Fixed
- Removed small crash for small grid and big number of processors

### Changed

### Known Bugs

- On gcc 6.1 the project does not compile
- Distributed grids on 1D do not work


## OpenFPM 0.4.0 - May 2016

### Added
- Grid with periodic boundary conditions
- VTK Writer for distributed vector, now is the default writer
- Installation of linear algebra packages
- More user friendly installation (No environment variables to add in your bashrc, installation report less verbose)

### Fixed
- GPU compilation
- PARMetis automated installation
- Critical Bug in getCellList, it was producing Celllist with smaller spacing

### Changed


## OpenFPM 0.3.0 - Apr 2016

### Added
- Molacular Dynamic example
- addUpdateCell list for more optimal update of the cell list instead of recreate the CellList

### Fixed
- Nothing to report

### Changed
- Eliminated global_v_cluster, init_global_v_cluster, delete_global_v_cluster, 
  substituted by 
  create_vcluster, openfpm_init, openfpm_finalize
- CartDecomposition parameter for the distributed structures is now optional
- template getPos<0>(), substituted by getPos()

## OpenFPM 0.2.1 - Apr 2016

### Changed
- GoogleChart name function changed: AddPointGraph to AddLinesGraph and AddColumsGraph to AddHistGraph

## OpenFPM 0.2.0 - Mar 2016
### Added
- Added Load Balancing and Dynamic Load Balancing on Beta
- PSE 1D example with multiple precision
- Plot example for GoogleChart plotting
- Distributed data structure now support 128bit floating point precision (on Beta)

### Fixed
- Detection 32 bit system and report as an error
- Bug in rounding off for periodic boundary condition

### Changed
- Nothing to report

## OpenFPM 0.1.0 - Feb 2016
### Added
- PSE 1D example
- Cell list example
- Verlet list example
- Kickstart for OpenFPM_numeric
- Automated dependency installation for SUITESPRASE EIGEN OPENBLAS(LAPACK)


### Fixed
- CRITICAL BUG in periodic bondary condition
- BOOST auto updated to 1.60
- Compilation with multiple .cpp files

### Changed
- Nothing to report





