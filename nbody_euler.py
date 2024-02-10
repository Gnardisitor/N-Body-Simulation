""" N-Body Simulation

This script allows the user to simulate any section of the solar system
and displays the final orbits, the initial and final position of the
planet in a final graph.

This version currently only contains the Euler method.

This script requires that numpy, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.1.0
"""

# Import required libraries
import math									# Used only for the function sqrt()
import numpy as np 							# Used for better arrays used as vectors
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

    Methods
    -------
    addAcc()
		Calculates the acceleration of the body from another planetary body
    resetAcc()
    	Resets the planetary body's acceleration to a zero vector
    move()
    	Calculates the body's velocity and position vectors using the Euler method
    """

	def __init__(self, nasa_id, date_i, date_f):
		""" 
		Create the initial required variables for a planetary body

		Parameters
		----------
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		date_i : str
			String used for initial date for JPL Horizons
		date_f : str
			String used for final date for JPL Horizons
		"""

		# Sets arrays for the names, colors, and masses for each planetary body in the solar system
		nameList = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])
		colorList = np.array(["xkcd:goldenrod", "xkcd:grey", "xkcd:dark orange", "xkcd:clear blue", "xkcd:rust", "xkcd:light brown", "xkcd:tan", "xkcd:sea blue", "xkcd:grey blue"])
		massList = np.array([1.989e30, 3.301e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.683e26, 8.681e25, 1.024e26])

		# Sets the qualitative properties of the body from the predefined arrays from the id
		self.mass = massList[nasa_id]
		self.color = colorList[nasa_id]
		self.name = nameList[nasa_id]

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

	def addAcc(self, body):
		"""
		Add an additional gravitational force upon the planetary body

		Parameters
		----------
		body : Body()
			The target planetary body used for 
		"""

		vector = (body.position - self.position) * 1.496e11										# Measured in m
		r = math.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))					# Measured in N
		self.acceleration = self.acceleration + ((6.67e-11 * body.mass * vector) / (r ** 3))	# Measured in m/s^2

	def resetAcc(self):
		""" Resets the body's acceleration vector to a zero vector """

		self.acceleration = np.zeros(3)

	def move(self, h):
		"""
		Calculate the body's velocity and position vectors using the position Euler method

		Parameters
		----------
		h : float
			A float representing the time steps used for the Euler method (d)
		"""

		# Calculate the steps of the position Verlet method
		self.acceleration = self.acceleration * float((86400 ** 2) / 1.496e11)	# Changes from m/s^2 to AU/d^2
		self.velocity = self.velocity + (h * self.acceleration)					# Calculate velocity at n + 1 using acceleration at n
		self.position = self.position + (h * self.velocity)						# Calculate position at n + 1

		# Add the new position to the history array of the planetary body
		self.history_x = np.append([self.history_x], self.position[0])
		self.history_y = np.append([self.history_y], self.position[1])
		self.history_z = np.append([self.history_z], self.position[2])
		self.history = np.array([self.history_x, self.history_y, self.history_z])

class System():
	"""
	A class which acts as the API to get the initial positions and velocities
	of the planetaries bodies

    Attributes
    ----------
    n : int
        An integer representing the number of planets in the solar system
    h : float
		A float representing the time steps that the system uses for physics (d)
	bodies : np.array()
		An array used to store all the planetary body objects included in the system
    """

	def __init__(self, n, h, year, month, day):
		"""
		Create the required variables for the system

		Parameters
		----------
		n : int
        	An integer representing the number of planets in the solar system
    	h : float
			A float representing the time steps that the system uses for physics (d)
		year : int
		    An integer used to represent the initial year of the simulation
		month : int
		    An integer used to represent the initial month of the simulation
		day : int
			An integer used to represent the initial day of the simulation
		"""

		# Set time interval and number of planets for the simulation
		self.n = n
		self.h = h

		# Create initial and final date for the Horizons API to use
		self.date_i = "{}-{}-{}".format(year, month, day)
		self.date_f = "{}-{}-{}".format(year, month, day + 1)

		# Add all the planetary bodies to the system's array
		self.bodies = np.array([])
		for i in range(self.n+1):
			self.bodies = np.append([self.bodies], Body(i, self.date_i, self.date_f))

	def update(self):
		for i in range(self.n+1):
			for body in self.bodies:
				if body.mass != self.bodies[i].mass:
					self.bodies[i].addAcc(body)
			self.bodies[i].move(self.h)
			self.bodies[i].resetAcc()


# Sets an array for the maximum sizes of the system depending on how many planets are used
sizeList = np.array([0, 0.5, 1.0, 1.5, 2.0, 6.0, 10.0, 20.0, 30.0])

# Asks the user for the year, month, and day for the initial date of the simulation
year = int(input("What initial year does the simulation begin in?\n"))
month = int(input("What initial month does the simulation begin in?\n"))
day = int(input("What initial day does the simulation begin in?\n"))

# Asks the use for the number of planets wanted and initiates the solar system with a time step of 1 day (86400s)
n = int(input("How many planets do you want?\n"))
solar = System(n, 1.0, year, month, day)

# Initialize the graph
fig, ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
ax.set_xlim((-sizeList[n], sizeList[n]))
ax.set_ylim((-sizeList[n], sizeList[n]))
ax.set_zlim((-sizeList[n], sizeList[n]))

# Loops simulation for n days
n = int(input('How many days do you want to simulate?\n'))
for t in range(n):
	# Update the positions of the bodies in the solar system
	solar.update()

	# Prints the position of every planetary body and the frame number for debugging
	for body in solar.bodies:
		print(f'The position of {body.name} is {body.position} AU.')
	print(f'Frame {t+1} drawn!')	

# Plots all the positions of all the planetary bodies
for body in solar.bodies:
	ax.plot(*body.history, marker=".", markersize=1, alpha=0.2, color=body.color)
	first_history = np.array([body.history_x[0], body.history_y[0], body.history_z[0]])
	ax.plot(*first_history, marker="o", markersize=2, alpha=0.5, color=body.color)
	last = body.history_x.size - 1
	last_history = np.array([body.history_x[last], body.history_y[last], body.history_z[last]])
	ax.plot(*last_history, marker="o", markersize=3, alpha=1.0, color=body.color)

# Displays the final graph at the end
plt.show()