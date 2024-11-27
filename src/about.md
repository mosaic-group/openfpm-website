# About OpenFPM

OpenFPM is an open-source C++ framework for parallel particles-only and hybrid particle-mesh codes. OpenFPM is intended as a successor to the discontinued PPM Library [1](https://www.sciencedirect.com/science/article/pii/S002199910500505X),[2](https://www.sciencedirect.com/science/article/pii/S2210983815002746).

Please cite with:
```c
@article{INCARDONA2019155,
title = {OpenFPM: A scalable open framework for particle and particle-mesh codes on parallel computers},
journal = {Computer Physics Communications},
volume = {241},
pages = {155-177},
year = {2019},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2019.03.007},
url = {https://www.sciencedirect.com/science/article/pii/S0010465519300852},
author = {Pietro Incardona and Antonio Leo and Yaroslav Zaluzhnyi and Rajesh Ramaswamy and Ivo F. Sbalzarini},
```

## Publications OpenFPM: core
- Incardona P, Gupta A, Yaskovets S, Sbalzarini IF, [A portable C++ library for memory and compute abstraction on multi-core CPUs and GPUs](https://onlinelibrary.wiley.com/doi/10.1002/cpe.7870),  Concurrency and Computation Practice and Experience 2023
- Singh, A., Incardona, P. & Sbalzarini, I.F., [A C++ expression system for partial differential equations enables generic simulations of biological hydrodynamics](https://link.springer.com/article/10.1140/epje/s10189-021-00121-x), The European Physical Journal E, 2021


## Publications OpenFPM: applications
- A. Salman et al., [Active Freedericksz Transition in Active Nematic Droplets](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.041002), Physical Review X, 2024
- Zoeller, C. and Adams, N. A. and Adami, S., [Beam-shaping in laser-based powder bed fusion of metals: A computational analysis of point-ring intensity profiles](https://www.sciencedirect.com/science/article/pii/S2214860424004482), Additive Manufacturing, 2024
- Schulze, Lennart J. and Veettil, Sachin K. T. and Sbalzarini, Ivo F., [A high-order fully Lagrangian particle level-set method for dynamic surfaces](https://www.sciencedirect.com/science/article/pii/S0021999124005102), Journal of Computational Physics, 2024
- Wimmer A, Panzer H, Zoeller C, Adami A, Adams N A & Zaeh M F, [Experimental and numerical investigations of the hot cracking susceptibility during the powder bed fusion of AA 7075 using a laser beam](https://link.springer.com/article/10.1007/s40964-023-00523-7), Progress in Additive Manufacturing, 2023
- A. Singh, P. H. Suhrcke, P. Incardona, I. F. Sbalzarini, [A numerical solver for active hydrodynamics in three dimensions and its application to active turbulence](https://pubs.aip.org/pof/article/35/10/105155/2919100/A-numerical-solver-for-active-hydrodynamics-in), Physics of Fluids, 2023
- Singh, A., Foggia, A., Incardona, P. et al, [A Meshfree Collocation Scheme for Surface Differential Operators on Point Clouds](https://link.springer.com/article/10.1007/s10915-023-02313-3), Journal of Scientific Computing, 2023
- C. Zöller, N.A. Adams, S. Adami, [Numerical investigation of balling defects in laser-based powder bed fusion of metals with Inconel 718](https://www.sciencedirect.com/science/article/pii/S2214860423002713), Additive Manufacturing, 2023 
- Geara, S., Martin, S., Adami, S. et al. [SPH 3D simulation of jet break-up driven by external vibrations](https://link.springer.com/article/10.1007/s40571-023-00624-8), Cumputational Particle Mechanics, 2023
- Singh, Abhinav and Vagne, Quentin and Julicher, Frank and Sbalzarini, Ivo F, [Spontaneous flow instabilities of active polar fluids in three dimensions](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.L022061), Physical Review Research, 2023
- Veettil, Sachin Krishnan Thekke and Zavalani, Gentian and Acosta, Uwe Hernandez and Sbalzarini, Ivo F. and Hecht, Michael, [Global Polynomial Level Sets for Numerical Differential Geometry of Smooth Closed Surfaces](https://epubs.siam.org/doi/10.1137/22M1536510), SIAM Journal on Scientific Computing, 2023
- C. Zöller, N.A. Adams, S. Adami, [A partitioned continuous surface stress model for multiphase smoothed particle hydrodynamics](https://www.sciencedirect.com/science/article/pii/S0021999122007793), Journal of Computational Physics, 2023
- Maddu Suryanarayana, Cheeseman Bevan L., Sbalzarini Ivo F. and Müller Christian L., [Stability selection enables robust learning of differential equations from limited noisy data](https://royalsocietypublishing.org/doi/10.1098/rspa.2021.0916), Proceedings Of The Royal Society A
- Stoyanovskaya, O.P., Grigoryev, V.V., Suslenkova, A.N., Davydov, M.N., Snytnikov, N.V., [Two-Phase Gas and Dust Free Expansion: Three-Dimensional Benchmark Problem for CFD Codes](https://www.mdpi.com/2311-5521/7/2/51), Fluids 2022
- Nesrine Khouzami, Friedrich Michel, Pietro Incardona, Jeronimo Castrillon, Ivo F. Sbalzarini, [Model-based autotuning of discretization methods in numerical simulations of partial differential equations](https://www.sciencedirect.com/science/article/pii/S1877750321001563), Journal of Computational Science, 2022
- Wimmer A, Yalvac B, Zoeller C, Hofstaetter F, Adami S, Adams NA, Zaeh MF., [Experimental and Numerical Investigations of In Situ Alloying during Powder Bed Fusion of Metals Using a Laser Beam](https://www.mdpi.com/2075-4701/11/11/1842), Metals. 2021
- Picó, J., Vignoni, A., Boada, Y., [Stochastic Differential Equations for Practical Simulation of Gene Circuits](https://link.springer.com/protocol/10.1007/978-1-0716-1032-9_2), Methods in Molecular Biology, 2021
- Gupta, Aryaman and Incardona, Pietro and Aydin, Ata Deniz and Gumhold, Stefan and Gunther, Ulrik and Sbalzarini, Ivo F., [An Architecture for Interactive In Situ Visualization and its Transparent Implementation in OpenFPM](https://dl.acm.org/doi/abs/10.1145/3426462.3426472), ISAV'20 In Situ Infrastructures for Enabling Extreme-Scale Analysis and Visualization, ACM, 2020
- Daniel R. Weilandt, Vassily Hatzimanikatis, [Particle-Based Simulation Reveals Macromolecular Crowding Effects on the Michaelis-Menten Mechanism](https://www.sciencedirect.com/science/article/pii/S0006349519305065), Biophysical Journal, 2019
- A. Gupta et al. [A Proposed Framework for Interactive Virtual Reality In Situ Visualization of Parallel Numerical Simulations](https://ieeexplore.ieee.org/document/8944368), 2019 IEEE 9th Symposium on Large Data Analysis and Visualization (LDAV), 2019

## License

- [BSD 3-Clause](https://github.com/mosaic-group/openfpm/blob/master/LICENSE.md)
