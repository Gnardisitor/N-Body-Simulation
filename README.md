N-Body Simulation

This script allows the user to simulate any section of the solar system
the use wants tracing orbits and be able to save the result.

It can save the resulting graph as a pdf and png under the name
system.png or system.pdf.

This version currently only contains the Euler method to approximate the
planetary bodies' position vectors and velocity vectors. Next versions
will also contain the Leapfrog integration method as well as the Runge-Kutta
fourth order method (RK4).

This script requires that numpy, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.0.0
