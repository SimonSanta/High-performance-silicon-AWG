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
* Python for data processing, optimizations and overall characteristics computation.

## Detailed Description

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
4. Simulations and optimization results
5. Overall characteristics computation and final parameters values

__Results:__ The study demonstrate improvements compared to simple transition
shapes usually used, evaluate the device overall performances and computed the final 
parameters values for an operating device.

## Results
## Metrics
![measurement setup and metrics](https://user-images.githubusercontent.com/48290004/152970739-d6fd7162-1671-415f-a1b2-7e0ebd06c3da.PNG)*Fig - Layout and metrics*

### FEM simulation
FEM meshing        |   waveguide fundamental mode
:-------------------------:|:-------------------------:
![singlemodemesh](https://user-images.githubusercontent.com/48290004/153052357-e503d7aa-8c0c-4eb0-9fe9-35e4ecddbca2.PNG)|![singlemodewg](https://user-images.githubusercontent.com/48290004/153052402-260f63e9-fcb6-4361-a256-b6c9e031b926.PNG)

*Fig2. - FEM used for waveguide mode solving in the scope of the project*

### FDTD simulation
Finite-difference time-domain (or FDTD) is a method directly derived from Maxwell’s curl
equations. It is one of the most famous numerical methods for solving for fields in mediums. FDTD works even in the case
of complex geometries with quickly varying envelope or backward reflections, something
that cannot be done by some other famous simulation methods such as the BPM, which
suppose a slowly varying envelope. This is mainly the reason why it was chosen for the
computation of the field across the irregular interface domain.

FDTD simulation allows to take into account complicated 
![fdtdsimu2](https://user-images.githubusercontent.com/48290004/153053602-56d1e345-2168-469e-aac5-c1497ef1ec04.PNG)
*Fig3. - FDTD simulation used for fields propagation in the scope of the project*

### Optimization
Multiple optimization techniques are used along the project. Here the final results are presented for one case.

![optimization1](https://user-images.githubusercontent.com/48290004/153063913-e6048f7c-027c-4372-a462-1ebad3b59978.PNG)
*Fig4. - Taper first stage optimization final results*

### Overall characteristics
Step-by-step simulation methods and results for the overall characterictics computation. The sensitive interface optimized structure
was used at the boundary between the free-propagation region and the waveguide array. A unit power is used at the input beam.

![fulllayoutsimu](https://user-images.githubusercontent.com/48290004/152956658-46663b4c-e582-492f-ab08-50fbb472fec7.PNG)
*Fig5. - Overall simulation flowchart*

Step1
![input_beam](https://user-images.githubusercontent.com/48290004/153057264-a468037e-aa3b-4153-a48f-45556166f906.PNG)
*Fig6. - Input beam*

Step3
![coupling_to_array](https://user-images.githubusercontent.com/48290004/153057596-8195c8cf-07fb-4af4-bd28-3d1aeb01d984.PNG)
*Fig7. - Coupling to array*

Step4
![field_in_wgs](https://user-images.githubusercontent.com/48290004/153057604-41660992-be11-4096-8b5c-47456843b0f6.PNG)
*Fig8. - Field in the waveguide array*

Step 6,7 and 8
![field_fpr2](https://user-images.githubusercontent.com/48290004/153057614-3e4b6c15-36a1-4adf-8397-83524525d0e1.PNG)
*Fig9. - Field out of array displayed on the top right picture and far in the second free-propagation region using FFT (far-field) on the bottom right picture*

### Final values
Interface optimal design and values         |   AWG final design parameters and characteristics
:-------------------------:|:-------------------------:
![final_optimal_parameters](https://user-images.githubusercontent.com/48290004/152979609-c84fd644-78e4-4158-b415-edabddd1b4a8.PNG)  |  ![final_design_parameters](https://user-images.githubusercontent.com/48290004/152966115-fcc6e55c-6dc0-4800-a982-51ed8bb8be1c.PNG)

*Fig10. - Final values for interface optimization and global parameters*

*Most of the files, PDF and Powerpoint presentation as well as thesis paper on Github.*
