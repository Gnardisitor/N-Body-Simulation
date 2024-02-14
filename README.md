N-Body Simulation

This script allows the user to simulate any section of the solar system,
trace orbits, and be able to save the result.

There are two new optional python files now which run much faster and display
a final graph either using Euler or position Verlet (Leapfrog).

This script requires that numpy, tqdm, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
pip install tqdm
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.1.2
