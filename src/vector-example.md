<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$']]}});
</script>
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS_HTML">
</script>
For detailed documentation of the OpenFPM sources, including the examples, see the
[online Doxygen documentation](https://ppmcore.mpi-cbg.de/doxygen/openfpm/).

<div id="ex0" markdown="1">
##Vector 0: Simple vector initialization
This example show several basic functionalities of the distributed vector `vector_dist`. 
The distributed vector is a set of particles in an N-dimensional space.
In this example it is shown how to:

- Initialize the library
- Create a Box that defines the domain
- An array that defines the boundary conditions
- A Ghost object that will define the extension of the ghost part in physical units

 _The source code of the example [Vector/0_simple/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/0_simple/main.cpp). The full doxygen documentation [Vector_0_simple](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_0_simple.html)_.

See also our video lectures dedicated to this topic [Video 1](https://ppmcore.mpi-cbg.de/upload/video/Lesson1-1.mp4), [Video 2](https://ppmcore.mpi-cbg.de/upload/video/Lesson1-2.mp4)
<div style="clear:both;"></div>
<br></div>

<div id="ex1" markdown="1">
##Example 1: Vector Ghost layer
<img class="floatright" src="https://ppmcore.mpi-cbg.de/web/images/examples/after_ghost_get.jpg" style="width: 300px; height: 300px;">

This example shows the properties of `ghost_get` and `ghost_put` - functions 
that synchronize the ghosts layer for a distributed vector `vector_dist`.

In this example it is shown how to:

- Iterate `vector_dist` via `getDomainIterator` 
- Redistribute the particles in `vector_dist` according to the underlying domain decomposition via `map`
- Synchronize the ghost layers in the standard way
- `NO_POSITION`, `KEEP_PROPERTIES` and `SKIP_LABELLING` options of the `ghost_get` function
- Propagate the data from ghost to non-ghost particles via `ghost_put`

 _The source code of the example [Vector/1_ghost_get_put/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/1_ghost_get_put/main.cpp). The full doxygen documentation [Vector_1_ghost_get](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_1_ghost_get.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex2" markdown="1">
##Example 2: Cell-lists and Verlet-lists

This example shows the properties of `ghost_get` and `ghost_put` - functions 
that synchronize the ghosts layer for a distributed vector `vector_dist`.

Key points:

- How to utilize the grid iterator `getGridIterator`, to create a grid-like particle domain
- Two principal types of fast neighbor lists: cell-list `getCellList` and Verlet-list `getVerlet` for a distributed vector `vector_dist`
- `CELL_MEMFAST`, `CELL_MEMBAL` and `CELL_MEMMW` variations of the cell-list, with different memory requirements and computations costs
- Iterating through the neighboring particles via `getNNIterator` of cell-list and Verlet-list

 _The source code of the example [Vector/1_celllist/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/1_ghost_get_put/main.cpp). The full doxygen documentation [Vector_1_celllist](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_1_celllist.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex3" markdown="1">
##Example 3: GPU vector
<img class="floatright" src="../img/examples/1_gpu_first_step.png" style="width: 300px; height: 300px;">

This example shows how to create a vector data-structure with `vector_dist_gpu` to access a `vector_dist`-alike data structure from GPU accelerated computing code.

Key points:

- How to convert the source code from using `vector_dist` to `vector_dist_gpu` and how it influences the memory layout of the data structure
- Oflloading particle position `hostToDevicePos` and particle property `hostToDeviceProp` data from CPU to GPU
- Lanuching a CUDA-like kernel with `CUDA_LAUNCH` and automatic subdivision of a computation loop into workgroups/threads via `getDomainIteratorGPU` or manually specifying the number of workgroups and the number of threads in a workgroup
- Passing the data-structures to a CUDA-like kernel code via `toKernel`
- How to use `map` with the option `RUN_DEVICE` to redistribute the particles directly on GPU, and `ghost_get` with `RUN_DEVICE` option to fill ghost particles directly on GPU
- How to detect and utilize RDMA on GPU to get the support of CUDA-aware MPI implementation to work directly with device pointers in communication subroutines

 _The source code of the example [Vector/1_gpu_first_step/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/1_gpu_first_step/main.cpp). The full doxygen documentation [Vector_1_gpu_first_step](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_1_gpu_first_step.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex4" markdown="1">
##Example 4: HDF5 Save and load

This example show how to save and load a vector to/from the parallel file format HDF5.

Key points:

- How to save the position/property information of the particles `vector_dist` into an _.hdf5_ file via `save`
- How to load the position/property information of the particles `vector_dist` from an _.hdf5_ file via `load`

 _The source code of the example [Vector/1_HDF5_save_load/main.cpp](https://github.com/mosaic-group/openfpm/tree/master/example/Vector/1_HDF5_save_load/main.cpp). The full doxygen documentation [Vector_1_HDF5](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_1_HDF5.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex5" markdown="1">
##Example 5: Vector expressions

This example shows how to use vector expressions to apply mathematical operations and functions on particles. 
The example also shows to create a point-wise applicable function   
$$ A_q e^{\frac{|x_p-x_q|^2}{\sigma}} $$

where
$A_q$ is the property $A$ of particle $q$, $x_p, x_q$ are positions of particles $p, q$ correspondingly.


Key points:

- Setting an alias for particle properties via `getV` of `particle_dist` to be used within an expression 
- Composing expressions with scalar particle properties 
- Composing expressions with vector particle properties. The expressions are 1) applied point-wise; 2) used to create a component-wise multiplication via `*`; 3) scalar product via `pmul`; 4) compute a norm `norm`; 5) perform square root operation `sqrt`
- Converting `Point` object into an expression `getVExpr` to be used with vector expressions
- Utilizing ` operator=` and the function `assign` to assing singular or multiple particle properties per iteration through particles
- Constructing expressions with `applyKernel_in` and `applyKernel_in_gen` to create kernel functions called at particle locations 
for all the neighboring particles, e.g. as in SPH

$$\sum_{q = Neighborhood(p)}  A_q D^{\beta}ker(x_p,x_q) V_q $$

 _The source code of the example [Vector/2_expressions/main.cpp](https://github.com/mosaic-group/openfpm/tree/master/example/Vector/2_expressions/main.cpp). The full doxygen documentation [Vector_2_expression](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_2_expression.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex6" markdown="1">
##Example 6: Molecular Dynamics with Lennard-Jones potential (Cell-List)

This example shows a simple Lennard-Jones molecular dynamics simulation in a stable regime. 
The particles interact with the interaction potential   
$$ V(x_p,x_q) = 4( (\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6  ) $$

$A_q$ is the property $A$ of particle $q$, $x_p, x_q$ are positions of particles $p, q$ correspondingly, $\sigma$ is a free parameter, $r$ is the distance between the particles.

Key points:

- Reusing memory allocated with `getCellList` for the subsequent iterations via `updateCellList`
- Utilizing `CELL_MEMBAL` with `getCellList` to minimize memory footprint
- Performing 10000 time steps using symplectic Verlet integrator

$$ \vec{v}(t_{n}+1/2) = \vec{v}_p(t_n) + \frac{1}{2} \delta t \vec{a}(t_n) $$

$$ \vec{x}(t_{n}+1) = \vec{x}_p(t_n) + \delta t \vec{v}(t_n+1/2) $$

$$ \vec{v}(t_{n+1}) = \vec{v}_p(t_n+1/2) + \frac{1}{2} \delta t \vec{a}(t_n+1) $$

- Producing a time-total energy 2D plot with `GoogleChart`

 _The source code of the example [Vector/3_molecular_dynamic/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/3_molecular_dynamic/main.cpp). The full doxygen documentation [Vector_3_md_dyn](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_3_md_dyn.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex7" markdown="1">
##Example 7: Molecular Dynamics with Lennard-Jones potential (Verlet-List)

The physical model in the example is identical to [Molecular Dynamics with Lennard-Jones potential (Cell-List)](#ex6). Please refer to it for futher details.
Key points:

- Due to the computational cost of updating Verlet-list, _r_cut + skin_ cutoff distance is used
such that the Verlet-list has to be updated once in 10 iterations via `updateVerlet`
- As Verlet-lists are constructed based on local particle id's, which would be invalidated by `map` or `ghost_get` ,`map` is called every 10 time-step, and `ghost_get` is used with `SKIP_LABELLING` option to keep old indices every iteration

 _The source code of the example [Vector/3_molecular_dynamic/main_vl.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/3_molecular_dynamic/main_vl.cpp). The full doxygen documentation [Vector_3_md_vl](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_3_md_vl.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex15" markdown="1">
##Example 15: Molecular Dynamics with Lennard-Jones potential (Symmetric Verlet-List)

This example is an extension to [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7). It shows how better performance can be achieved for symmetric interaction models with symmetric Verlet-list compared to the standard Verlet-list.
Key points:

- Computing the interaction for particles _p_, _q_ only once
- Propagate the data from potentially ghost particles _q_ to non-ghost particles in their corresponding domains via `ghost_put` with the operation `add_`
- Changing the prefactor in the subroutine of calculating the total energy as every pair of particles is visited once (as compared to two times before) 
- Updating Verlet-list once in 10 iterations via `updateVerlet` with 'VL_SYMMETRIC' flag

 _The source code of the example [Vector/5_molecular_dynamic_sym/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/5_molecular_dynamic_sym/main.cpp). The full doxygen documentation [Vector_5_md_vl_sym](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_5_md_vl_sym.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex16" markdown="1">
##Example 16: Molecular Dynamics with Lennard-Jones potential (Symmetric CRS Verlet-List)

This example is an extension to [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7) and [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex15). It shows how better performance can be achieved for symmetric interaction models with symmetric Verlet-list compared to the standard Verlet-list.
Key points:

- Computing the interaction for particles _p_, _q_ only once
- Propagate the data from potentially ghost particles _q_ to non-ghost particles in their corresponding domains via `ghost_put` with the operation `add_`
- Changing the prefactor in the subroutine of calculating the total energy as every pair of particles is visited once (as compared to two times before) 
- Updating Verlet-list once in 10 iterations via `updateVerlet` with 'VL_SYMMETRIC' flag

 _The source code of the example [Vector/5_molecular_dynamic_sym/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/5_molecular_dynamic_sym/main.cpp). The full doxygen documentation [Vector_5_md_vl_sym](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_5_md_vl_sym.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex8" markdown="1">
##Example 8: Molecular Dynamics with Lennard-Jones potential (GPU)

The physical model in the example is identical to [Molecular Dynamics with Lennard-Jones potential (Cell-List)](#ex6) and [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7). Please refer to those for futher details.
Key points:

- To get the particle index inside a CUDA-like kernel `GET_PARTICLE` macro is used to avoid overflow in the construction `blockIdx.x * blockDim.x + threadIdx.x`
- A primitive reduction function `reduce_local` with the operation `_add_` is used to get the total energy by summing energies of all particles.

 _The source code of the example [Vector/3_molecular_dynamic_gpu/main_vl.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/3_molecular_dynamic_gpu/main.cu). The full doxygen documentation [Vector_3_md_dyn_gpu](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_3_md_dyn_gpu.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex9" markdown="1">
##Example 9: Molecular Dynamics with Lennard-Jones potential (GPU optimized)

The physical model in the example is identical to [Molecular Dynamics with Lennard-Jones potential (Cell-List)](#ex6), [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7) and is based on [Molecular Dynamics with Lennard-Jones potential (GPU)](#ex8). Please refer to those for futher details.
Key points:

- To achieve coalesced memory access on GPU and to reduce cache load the particle indices are stored in cell-list in a sorted manner, i.e. particles with neighboring indices are located in the same cell-list. This is achieved by assigning new particle indices and storing them temporarily in `vector_dist`. 
- The sorted version of `vector_dist_gpu` is offloaded to GPU using `toKernel_sorted`. It uses `get_sort` instead of `get` to get a particle index in the cell-list neighborhood iterator `getNNIteratorBox`
- The sorted version of particle properties have to be merged to the original ones once the processing is done via `merge_sort` of `vector_dist` 

 _The source code of the example [Vector/3_molecular_dynamic_gpu_opt/main_vl.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/3_molecular_dynamic_gpu_opt/main_gpu.cu). The full doxygen documentation [Vector_3_md_dyn_gpu_opt](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_3_md_dyn_gpu_opt.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex10" markdown="1">
##Example 10: Molecular Dynamics with Lennard-Jones potential (Particle reordering)

The physical model in the example is identical to [Molecular Dynamics with Lennard-Jones potential (Cell-List)](#ex6), [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7). The example shows how reordering the data can significantly reduce the computational running time. 
Key points:

- The particles inside `vector_dist` are reordered via `reorder` following a Hilbert curve of order _m_ (here _m=5_) passing through the cells of $2^m \times 2^m \times 2^m$ (here, in 3D) cell-list
- It is shown that the frequency of reordering depends on the mobility of particles
- Wall clock time is measured of the function `calc_force` utilizing the object `timer` via `start` and `stop`

 _The source code of the example [Vector/4_reorder/main_data_ord.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/4_reorder/main_data_ord.cpp). The full doxygen documentation [Vector_4_reo](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_4_reo.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex11" markdown="1">
##Example 11: Molecular Dynamics with Lennard-Jones potential (Cell-list reordering)

The physical model in the example is identical to [Molecular Dynamics with Lennard-Jones potential (Cell-List)](#ex6), [Molecular Dynamics with Lennard-Jones potential (Verlet-List)](#ex7). The example shows how reordering the data can significantly reduce the computational running time. 
Key points:

- The cell-list cells are iterated following a Hilbert curve instead of a normal left-to-right bottom-to-top cell iteration (in 2D). The function `getCellList_hilb` of `vector_dist` is used instead of `getCellList`
- It is shown that for static or slowly moving particles a speedup of up to 10% could be achieved

 _The source code of the example [Vector/4_reorder/main_comp_ord.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/4_reorder/main_comp_ord.cpp). The full doxygen documentation [Vector_4_comp_reo](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_4_comp_reo.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex12" markdown="1">
##Example 12: Complex properties in Vector 1

This example shows how to use complex properties in the distributed vector `vector_dist`

Key points:

- Creating a distributed vector with particle properties: scalar, vector `float[3]`, `Point`, list of float `openfpm::vector<float>`, list of custom structures `openfpm::vector<A>` (where `A` is a user-defined type with no pointers), vector of vectors `openfpm::vector<openfpm::vector<float>>>`
- Redistribute the particles in `vector_dist` according to the underlying domain decomposition. Communicate only the selected particle properties via `map_list` (instead of communicating all `map`)
- Synchronize the ghost layers only for the selected particle properties `ghost_get`

 _The source code of the example [Vector/4_complex_prop/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/4_complex_prop/main.cpp). The full doxygen documentation [Vector_4_complex_prop](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_4_complex_prop.html)_.
<div style="clear:both;"></div>
<br></div>


<div id="ex13" markdown="1">
##Example 13: Complex properties in Vector 2

This example shows how to use complex properties in the distributed vector `vector_dist`

Key points:

- Creating a distributed vector with particle properties: scalar, vector `float[3]`, `Point`, list of float `openfpm::vector<float>`, list of custom structures `openfpm::vector<A>` (where `A` is a user-defined type with memory pointers inside), vector of vectors `openfpm::vector<openfpm::vector<float>>>`
- Enabling the user-defined type being serializable by `vector_dist` via 
      - `packRequest` method to indicate how many byte are needed to serialize the structure
      - `pack` method to serialize the data-structure via methods `allocate`, `getPointer` of `ExtPreAlloc` and method `pack` of `Packer` 
      - `unpack` method to deserialize the data-structure via method `getPointerOffset` of `ExtPreAlloc` and method `unpack` of `Unpacker`
      - `noPointers` method to inform the serialization system that the object has pointers
      - Constructing constructor, destructor and `operator=` to avoid memory leaks 

 _The source code of the example [Vector/4_complex_prop/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/4_complex_prop/main.cpp). The full doxygen documentation [Vector_4_complex_prop_ser](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_4_complex_prop_ser.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex14" markdown="1">
##Example 14: Multiphase Cell-lists and Verlet-lists

This example is an extension to [Example 2: Cell-lists and Verlet-lists](#ex2) and ()[]. It shows how to use multi-phase cell-lists and Verlet-list using multiple instances of `vector_dist`.
Key points:

- All the phases have to use the same domain decomposition, which is achieved by passing the decomposition of the first phase to the constructor of `vector_dist` of all the other phases.
- The domains have to be iterated individually via `getDomainIterator`, the particles redistributed via `map`, the ghost layers synchronized via `ghost_get` for all the phases `vector_dist`.
- Constructing Verlet-lists for two phases (_ph0_, _ph1_) with `createVerlet`, where for one phase _ph0_ the neighoring particles of _ph1_ are assigned in the Verlet-list. Cell-list of _ph1_ has to be passed to `createVerlet`
- Constructing Verlet-lists for multiple phases (_ph0_, _ph1_, _ph2_...) with `createVerletM`, where for one phase _ph0_ the neighoring particles of _ph1_, _ph2_... are assigned in the Verlet-list. Cell-list containing all of _ph1_, _ph2_... create with `createCellListM` has to be passed to `createVerletM`
- Iterating over the neighboring particles of a multiphase Verlet-list with `getNNIterator` with `get` being substituded by `getP` (particle phase) and `getV` (particle id)
- Extending example of the symmetric interaction for multiphase cell-lists and Verlet-lists via `createCellListSymM`, `createVerletSymM`

 _The source code of the example [Vector/4_multiphase_celllist_verlet/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/4_multiphase_celllist_verlet/main.cpp). The full doxygen documentation [Vector_4_mp_cl](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_4_mp_cl.html)_.
<div style="clear:both;"></div>
<br></div>

<div id="ex16" markdown="1">
##Example 16: Validation and debugging

This example shows how the flexibility of the library can be used to perform complex tasks for validation and debugging.
Key points:

- To get unique global id's of the particles the function `accum` of `vector_dist` is used, which returns prefix sum of local domain sizes $j<i$ for the logical processor $i$ out of $N$ total processors
- Propagate the data from potentially ghost particles _q_ to non-ghost particles in their corresponding domains via `ghost_put` with the operation `merge_`, that merges two `openfpm::vector` (ghost and non-ghost)

 _The source code of the example [Vector/6_complex_usage/main.cpp](https://github.com/mosaic-group/openfpm/blob/master/example/Vector/6_complex_usage/main.cpp). The full doxygen documentation [Vector_6_complex_usage](https://ppmcore.mpi-cbg.de/doxygen/openfpm/Vector_6_complex_usage.html)_.
<div style="clear:both;"></div>
<br></div>


<div style="clear:both;"/></div>
