# High-performance-silicon-AWG

__*High Performance Si AWG With Geometrically Improved Interface Between Slab And Waveguide Array*__,
*Sep 2020 – Sep 2021*

![fulllayoutsimu](https://user-images.githubusercontent.com/48290004/152956658-46663b4c-e582-492f-ab08-50fbb472fec7.PNG)*Fig1 - Overall simulation flowchart*

<!---<img src="https://user-images.githubusercontent.com/48290004/152956658-46663b4c-e582-492f-ab08-50fbb472fec7.PNG" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="1311" height="736" />--->

Master's Thesis in optoelectronics and silicon photonics in the scope of the T.I.M.E. Double Degree program between UCLouvain (Belgium) and Keio University (Japan).

__Tasks:__ Design of an optical (de)multiplexer (arrayed waveguide grating), aiming at both higher transmission performances and reducing the Si photonics drawbacks with respect to what is currently used in the industry.

Used in this context: 
* RSoft CAD Environment™ for circuit layout design
* FemSIM™ using Finite Element Method (FEM) for waveguide/slab mode solving
* FullWAVE™ using Finite-Difference Time-Domain (FDTD) method for light propagation at sensitive material interfaces
* Python for data processing, optimization and overall characteristics computations

## Detailed Explanation

__Framework:__ 
Arrayed Waveguide Gratings (AWG), also called PHASARS in the literature, are optical
communication devices which play a key role in Wavelength Division Multiplexing (WDM),
a technique considerably improving networks transmission capacity and flexibility. Taking
light as an input and by the successive use of Fourier Optics transformation as well as a
waveguide array, which provides the focusing and dispersive properties, the AWG are able to
operate as wavelength multiplexers, demultiplexers and routers.

Firstly built in silica which resulted
in too bulky structures and with the advance of Si photonics technology, Si was therefore
used for the miniaturization of AWGs. This has permitted to reach dimensions of one or
two order of magnitude smaller.

__Issues:__ Si AWG’s waveguides enhanced optical
confinement induces larger phase error and scattering/losses in transition regions. This is
typically resulting in deteriorated performances of crosstalk and insertion loss. The most
sensitive region located at the abrupt transition, with a high index difference between *Si* and
*SiO<sub>2</sub>*, between the planar slab region and the arrayed WGs region, which yields to optical
field mismatch in addition to scattering.

__Problem solving strategy__: 
1. Historical review, state-of-the-art study and industry benchmarks
2. New improvements to the structure
3. Step-by-step optimization techniques design 
4. Simulation, data acquisition and optimization
5. Overall characteristics computation and final results

__Results:__ The study demonstrate improvements compared to simple transition
shapes usually used, evaluate the device overall performances and computed the final 
parameters values for an operating device.

## Metrics
![measurement setup and metrics](https://user-images.githubusercontent.com/48290004/152970739-d6fd7162-1671-415f-a1b2-7e0ebd06c3da.PNG)*Fig1 - Overall simulation flowchart*

## Results
### FEM result example
Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![final_optimal_parameters](https://user-images.githubusercontent.com/48290004/152979609-c84fd644-78e4-4158-b415-edabddd1b4a8.PNG)  |  ![final_design_parameters](https://user-images.githubusercontent.com/48290004/152966115-fcc6e55c-6dc0-4800-a982-51ed8bb8be1c.PNG)*Fig1 - Final values for interface optimization and global parameters*

### FDTD result example
Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![final_optimal_parameters](https://user-images.githubusercontent.com/48290004/152979609-c84fd644-78e4-4158-b415-edabddd1b4a8.PNG)  |  ![final_design_parameters](https://user-images.githubusercontent.com/48290004/152966115-fcc6e55c-6dc0-4800-a982-51ed8bb8be1c.PNG)*Fig1 - Final values for interface optimization and global parameters*

### Overall characteristics
![fulllayoutsimu](https://user-images.githubusercontent.com/48290004/152956658-46663b4c-e582-492f-ab08-50fbb472fec7.PNG)*Fig1 - Overall simulation flowchart*

### Final values
Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![final_optimal_parameters](https://user-images.githubusercontent.com/48290004/152979609-c84fd644-78e4-4158-b415-edabddd1b4a8.PNG)  |  ![final_design_parameters](https://user-images.githubusercontent.com/48290004/152966115-fcc6e55c-6dc0-4800-a982-51ed8bb8be1c.PNG)*Fig1 - Final values for interface optimization and global parameters*

*Most of the files, PDF and Powerpoint presentation as well as thesis paper on Github.*
