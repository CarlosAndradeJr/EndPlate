# EndPlate
Design an end plate connection subject to two-way bending is not an easy task. In order to get a better estimate of the true force on each bolt it is necessary to satisfy the equilibrium (summation of forces and summation of moments equal to zero).

For this purpose it is necessary a basic plastic analysis performed with a nonlinear program. The linear strain field assumes that the bearing surface rigidly translates and rotates with reference to an axis, named the plastic. Assuming that a well defined no-tension constitutive law has been assigned to the bearing area (compression) and for each bolt a tension-only  behaviour is expected. 

The problem of finding the exact plastic neutral axis can be solved by an iterative procedure (Newton Raphson). The numerical procedure stops when the unbalance force/moments is lower than a specified tolerance. It may also happen that a convergence is impossible for the selected geometry and loads and a new design is required.

A summary of the whole procedure may be found in item 7.9 of Rugarli (2018) Steel Connection Analysis.
