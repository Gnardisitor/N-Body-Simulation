N-Body Simulation

This script allows the user to simulate any section of the solar system.

There are three different variants of the programs:
1. Using the Euler method
2. Using the Verlet (Leapfrog) method
3. Using the Runge-Kutta Fourth Order (RK4) method

This script requires that numpy, tqdm, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
pip install tqdm
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.2.0
