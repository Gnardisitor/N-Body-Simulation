""" N-Body Simulation

This script allows the user to simulate any section of the solar system
and displays the final orbits, the initial and final position of the
planet in a final graph.

This version uses the Runge-Kutta fourth order method (RK4).
* It currently also contains a hidden Euler method used for testing.

This script requires that numpy, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
pip install tqdm
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.2.0
"""

# Change how acceleration is calculated to remove reset acceleration
# Use *args to add as many accelerations as wanted at the end

# Import required libraries
import math									# Used only for the function sqrt()
import numpy as np 							# Used for better arrays used as vectors
from tqdm import tqdm						# Progress bar for terminal in python
import matplotlib.pyplot as plt 			# Used for graphing
from astroquery.jplhorizons import Horizons # Used to get positions and velocity vectors from NASA's JPL Horizons

class Body():
	"""
	A class which represents the planetary bodies

    Attributes
    ----------
    mass : float
		A float representing the planetary body's mass (kg)
	color : str
		A string representing the color of the marker for the planetary body in the graph (using the xkcd color survey)
	name : str
		A string representing the name of the planetary body for easier indentification
	position : np.array()
		A numpy array representing the position vector of the planetary body
	velocity : np.array()
		A numpy array representing the velocity vector of the planetary body
	acceleration : np.array()
		A numpy array representing the acceleration vector of the planetary body
	history_x : np.array()
		A numpy array representing all x positions of the planetary body

    Methods
    -------
    getAcc()
		Calculates the acceleration between the planetary body and another one
    """

	def __init__(self, nasa_id, date_i, date_f):
		""" 
		Create the initial required variables for a planetary body

		Parameters
		----------
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		date_i : str
			A string representing the initial date of the simulation
		date_f : str
			A string representing the final date of the simulation
		"""

		# Sets arrays for the names, colors, and masses for each planetary body in the solar system
		NAMES = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])
		COLORS = np.array(["xkcd:goldenrod", "xkcd:grey", "xkcd:dark orange", "xkcd:clear blue", "xkcd:rust", "xkcd:light brown", "xkcd:tan", "xkcd:sea blue", "xkcd:grey blue"])
		MASSES = np.array([1.989e30, 3.301e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.683e26, 8.681e25, 1.024e26])

		# Sets the qualitative properties of the body from the predefined arrays from the id
		self.mass = MASSES[nasa_id]
		self.color = COLORS[nasa_id]
		self.name = NAMES[nasa_id]

		# Initialize the basic vectors of the planetary bodies
		if nasa_id != 0:
			# Use JPL Horizons to initialize the position and velocity vectors of the planets except for the sun
			bodyData = Horizons(id=nasa_id, location="@sun", epochs={'start': date_i,'stop': date_f,'step': '1d'}).vectors().as_array()
			dataString = str(bodyData[0])
			vectors = dataString.split(", ")
			self.position = np.array([float(vectors[3]), float(vectors[4]), float(vectors[5])])
			self.velocity = np.array([float(vectors[6]), float(vectors[7]), float(vectors[8])])
		else:
			self.position = np.zeros(3)
			self.velocity = np.zeros(3)
		self.acceleration = np.zeros(3)

		# Create an array used to store all the body's position history for the simulation
		self.history_x = np.array([])
		self.history_y = np.array([])
		self.history_z = np.array([])

	def getAcc(self, body):
		"""
		Calculates the gravitational acceleration between the body and another planetary body

		Parameters
		----------
		body : Body()
			The target planetary body used for the calculation of gravity

		Returns
		-------
		np.array()
			The numpy array representing the acceleration from gravity
		"""

		vector = (body.position - self.position) * 1.496e11						# Measured in m
		r = math.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))	# Measured in N
		acc = ((6.67e-11 * body.mass * vector) / (r ** 3))						# Measured in m/s^2
		return acc * ((86400.0 ** 2.0) / 1.496e11)								# Measured in AU/d^2

class System():
	"""
	A class which represents the system in which the planetary bodies are contained

    Attributes
    ----------
	date_i : str
		A string representing the initial date of the simulation
	date_f : str
		A string representing the final date of the simulation
    n : int
        An integer representing the number of planets in the solar system
    h : float
		A float representing the time steps that the system uses for physics (d)
	bodies : np.array()
		An array used to store all the planetary body objects included in the system
	
	Methods
    -------
    setAcc()
		Calculates and sets the acceleration for a planetary body in the system
	f()
		Calculates the derivative of the given position and velocity for RK4
	rk4()
		Calculates the positions and velocities using the Runge-Kutta Fourth Order method
	update()
		Updates all the planetary bodies in the system for a single time interval h
    """

	def __init__(self, n, h, year):
		"""
		A class which acts as the API to get the initial positions and velocities
		of the planetaries bodies

		Parameters
		----------
		date_i : str
			A string representing the initial date of the simulation
		date_f : str
			A string representing the final date of the simulation
		n : int
			An integer representing the number of planets in the solar system
		h : float
			A float representing the time steps that the system uses for physics (d)
		bodies : np.array()
			An array used to store all the planetary body objects included in the system
		"""

		# Set time interval and number of planets for the simulation
		self.n = n
		self.h = h

		# Create initial and final date for the Horizons API to use
		self.date_i = f"{year}-01-01"
		self.date_f = f"{year}-01-02"

		# Add all the planetary bodies to the system's array
		self.bodies = np.array([])
		for i in range(self.n+1):
			self.bodies = np.append(self.bodies, Body(i, self.date_i, self.date_f))

	def setAcc(self, nasa_id):
		"""
		Calculates and sets the acceleration for a planetary body in the system

		Parameters
		----------
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		"""

		# Create temporary variable
		temp_acc = np.zeros(3)

		# Get all the accelerations from all the planets
		for body in self.bodies:
			if body.mass != self.bodies[nasa_id].mass:
				temp_acc = temp_acc + self.bodies[nasa_id].getAcc(body)

		# Change acceleration value
		self.bodies[nasa_id].acceleration = temp_acc

	def f(self, y, k, nasa_id):
		"""
		Calculates the derivative of the given position and velocity for RK4

		Parameters
		----------
		y : np.array()
			The numpy array representing planetary body position and velocity
		k : np.array()
			The numpy array representing the k value added to y for RK4
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		"""
		
		# Save initial position
		init_pos = self.bodies[nasa_id].position

		# Replace with new position
		self.bodies[nasa_id].position = init_pos + k[0]

		# Compute the new 
		self.setAcc(nasa_id)
		new_vel = y[0] + k[1]
		new_acc = self.bodies[nasa_id].acceleration

		# Reset vectors and return values
		self.bodies[nasa_id].position = init_pos
		self.bodies[nasa_id].velocity = y[0]
		self.bodies[nasa_id].acceleration = y[1]
		return np.array([new_vel, new_acc])

	def rk4(self, nasa_id):
		"""
		Calculates the positions and velocities using the Runge-Kutta Fourth Order method for a body

		Parameters
		----------
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		"""

		# Set y as velocity and acceleration vectors (derivatives of position and velocity)
		self.setAcc(nasa_id)
		y = np.array([self.bodies[nasa_id].velocity, self.bodies[nasa_id].acceleration])

		# Calculate 4 different Eulers at different times
		k1 = y									# k1 is vel & acc
		k2 = self.f(y, self.h*(k1/2), nasa_id)	# Finds k2 using k1
		k3 = self.f(y, self.h*(k2/2), nasa_id)	# Finds k3 using k2
		k4 = self.f(y, self.h*k3, nasa_id)		# Finds k4 using k3

		# Do the weighted average of k1, k2, k3, k4
		self.bodies[nasa_id].position = self.bodies[nasa_id].position + ((self.h/6) * (k1[0] + (2 * k2[0]) + (2 * k3[0]) + k4[0]))
		self.bodies[nasa_id].velocity = self.bodies[nasa_id].velocity + ((self.h/6) * (k1[1] + (2 * k2[1]) + (2 * k3[1]) + k4[1]))

		# Add new position vector to the history arrays
		self.bodies[nasa_id].history_x = np.append(self.bodies[nasa_id].history_x, self.bodies[nasa_id].position[0])
		self.bodies[nasa_id].history_y = np.append(self.bodies[nasa_id].history_y, self.bodies[nasa_id].position[1])
		self.bodies[nasa_id].history_z = np.append(self.bodies[nasa_id].history_z, self.bodies[nasa_id].position[2])

	def update(self):
		"""
		Updates all the planetary bodies in the system for a single time interval h
		"""

		# Computes RK4 for every planetary body
		for nasa_id in range(self.n+1):
			self.rk4(nasa_id)

	# This is the old Euler method which I tested out
	"""
	def move(self):
		
		for body in self.bodies:

			# Computes using Euler method
			body.velocity = body.velocity + (self.h * body.acceleration)
			body.position = body.position + (self.h * body.velocity)

			# Add the new position to the history array of the planetary body
			body.history_x = np.append(body.history_x, body.position[0])
			body.history_y = np.append(body.history_y, body.position[1])
			body.history_z = np.append(body.history_z, body.position[2])

	def update(self):

		# Computes RK4 for every planetary body
		for nasa_id in range(self.n+1):
			self.setAcc(nasa_id)
		self.move()
	"""

# Sets an array for the maximum sizes of the system depending on how many planets are used
SIZES = np.array([0, 0.5, 1.0, 1.5, 2.0, 6.0, 10.0, 20.0, 30.0])

# Asks the user for the year, month, and day for the initial date of the simulation
year = int(input("What initial year does the simulation begin in?\n"))

# Asks the use for the number of planets wanted and initiates the solar system with a time step of 1 day (86400s)
n = int(input("How many planets do you want?\n"))
solar = System(n, 1.0, year)

# Initialize the graph
fig, ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
ax.set_xlim((-SIZES[n], SIZES[n]))
ax.set_ylim((-SIZES[n], SIZES[n]))
ax.set_zlim((-SIZES[n], SIZES[n]))

# Loops simulation for n days
n = int(input('How many days do you want to simulate?\n'))
bar = tqdm(total=n)
for t in range(n):
	# Update the positions of the bodies in the solar system
	solar.update()
	bar.update(1)
bar.close()

# Plots all the positions of all the planetary bodies
for body in solar.bodies:

	# Plot all positions of the body
	history = np.array([body.history_x, body.history_y, body.history_z])
	ax.plot(*history, marker=".", markersize=1, alpha=0.2, color=body.color)

	# Plot a bigger point for the position at the beginning of the simulation
	first_history = np.array([body.history_x[0], body.history_y[0], body.history_z[0]])
	ax.plot(*first_history, marker="o", markersize=2, alpha=0.5, color=body.color)

	# Plot a big point for the position at the end of the simulation
	last = body.history_x.size - 1
	last_history = np.array([body.history_x[last], body.history_y[last], body.history_z[last]])
	ax.plot(*last_history, marker="o", markersize=3, alpha=1.0, color=body.color)

# Displays the final graph at the end
plt.show()