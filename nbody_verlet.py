""" N-Body Simulation

This script allows the user to simulate any section of the solar system
and displays the final orbits, the initial and final position of the
planet in a final graph.

This version currently only contains the position Verlet Method.

This script requires that numpy, matplotlib, astroquery.jplhorizons and
any of their depencies be installed in the python runtime.

To install the libraries required, run the following commands in cmd:

pip install numpy
python -m pip install -U matplotlib
python -m pip install -U --pre astroquery[all]

Author : Dragos Bajanica
Version : 1.1.1
"""

# Import required libraries
import math									# Used only for the function sqrt()
import numpy as np 							# Used for better arrays used as vectors
import matplotlib.pyplot as plt 			# Used for graphing
from astroquery.jplhorizons import Horizons # Used to get positions and velocity vectors from NASA's JPL Horizons

class API():
	"""
	A class which acts as the API to get the initial positions and velocities of the planetaries bodies

    Attributes
    ----------
    date_i : str
        A formatted string of the initial date by the user for the API
    date_f : str
		A formatted string of the initial date with one day added onto it

    Methods
    -------
    getPosition()
        Returns the position of a planetary body at the initial date
    getVelocity()
    	Returns the velocity of a planetary body at the initial date
    """

	def __init__(self, year, month, day):
		"""
		Create the initial and final date used by the API using the given parameters

		Parameters
		----------
		year : int
		    An integer used to represent the initial year of the simulation
		month : int
		    An integer used to represent the initial month of the simulation
		day : int
			An integer used to represent the initial day of the simulation
		"""

		self.date_i = "{}-{}-{}".format(year, month, day)
		self.date_f = "{}-{}-{}".format(year, month, day + 1)

	def getPosition(self, nasa_id):
		"""
		Get the initial position of the planetary body using the API

		Parameters
		----------
		nasa_id : int
		    The integer id used by NASA's JPL Horizons for each planet (1 to 8)
		
		Returns
		-------
		position : np.array()
			An array representing the initial position vector of the planetary body
		"""

		bodyData = Horizons(id=nasa_id, location="@sun", epochs={'start': self.date_i,'stop': self.date_f,'step': '1d'}).vectors().as_array()
		dataString = str(bodyData[0])
		vectors = dataString.split(", ")
		return np.array([float(vectors[3]), float(vectors[4]), float(vectors[5])])

	def getVelocity(self, nasa_id):
		""" 
		Get the initial velocity vector of the planetary body using the API

		Parameters
		----------
		nasa_id : int
		    The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		
		Returns
		-------
		velocity : np.array()
			An array representing the initial velocity vector of the planetary body
		"""

		bodyData = Horizons(id=nasa_id, location="@sun", epochs={'start': self.date_i,'stop': self.date_f,'step': '1d'}).vectors().as_array()
		dataString = str(bodyData[0])
		vectors = dataString.split(", ")
		return np.array([float(vectors[6]), float(vectors[7]), float(vectors[8])])

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

	def __init__(self, nasa_id):
		""" 
		Create the initial required variables for a planetary body

		Parameters
		----------
		nasa_id : int
			The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
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

	def move(self, h, system):
		"""
		Calculate the body's velocity and position vectors using the position Verlet method

		Parameters
		----------
		h : float
			A float representing the time steps used for the Verlet method (d)
		"""

		# Calculate the steps of the position Verlet method
		self.position = self.position + (1/2 * h * self.velocity)				# Calculate position at n + 1/2

		# Calculate the acceleration of the body
		for body in system.bodies:
				if body.mass != self.mass:
					self.addAcc(body)

		self.acceleration = self.acceleration * float((86400 ** 2) / 1.496e11)	# Changes from m/s^2 to AU/d^2
		self.velocity = self.velocity + (h * self.acceleration)					# Calculate velocity at n + 1 using acceleration at n + 1/2
		self.position = self.position + (1/2 * h * self.velocity)				# Calculate position at n + 1 using velocity at n + 1/2

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

	def __init__(self, n, h):
		"""
		Create the required variables for the system

		Parameters
		----------
		n : int
        	An integer representing the number of planets in the solar system
    	h : float
			A float representing the time steps that the system uses for physics (d)
		"""

		# Set time interval and number of planets for the simulation
		self.n = n
		self.h = h

		# Add all the planetary bodies to the system's array
		self.bodies = np.array([])
		for i in range(self.n+1):
			self.bodies = np.append([self.bodies], Body(i))

# Sets an array for the maximum sizes of the system depending on how many planets are used
sizeList = np.array([0, 0.5, 1.0, 1.5, 2.0, 6.0, 10.0, 20.0, 30.0])

# Asks the user for the year, month, and day for the initial date of the simulation
year = int(input("What initial year does the simulation begin in?\n"))
month = int(input("What initial month does the simulation begin in?\n"))
day = int(input("What initial day does the simulation begin in?\n"))
horizons = API(year, month, day)

# Asks the use for the number of planets wanted and initiates the solar system with a time step of 1 day (86400s)
n = int(input("How many planets do you want?\n"))
solar = System(n, 1.0)

# Sets the initial position and velocity vectors of each planetary body except for the sun
for i in range(n+1):
	if i != 0:
		solar.bodies[i].position = horizons.getPosition(i)
		print(f"The position of {solar.bodies[i].name} is {solar.bodies[i].position} AU.")
		solar.bodies[i].velocity = horizons.getVelocity(i)
		print(f"The velocity of {solar.bodies[i].name} is {solar.bodies[i].velocity} AU/d.")

# Initialize the graph
fig, ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
ax.set_xlim((-sizeList[n], sizeList[n]))
ax.set_ylim((-sizeList[n], sizeList[n]))
ax.set_zlim((-sizeList[n], sizeList[n]))

# Loops simulation for n days
n = int(input('How many days do you want to simulate?\n'))
for t in range(n):
	# Update the positions of the bodies in the solar system
	for body in solar.bodies:
		body.move(solar.h, solar)
		body.resetAcc()

		# Prints the position of every planetary body and the frame number for debugging
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