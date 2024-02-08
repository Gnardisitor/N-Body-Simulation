""" N-Body Simulation

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
"""

# Import required libraries
import math									# Used only for the function sqrt()
import numpy as np 							# Used for better arrays used as vectors
import matplotlib as mpl 					# Used as graphing library
import matplotlib.pyplot as plt 			# Used for graphing in 3D
from astroquery.jplhorizons import Horizons # Used to get positions and velocity vectors from NASA's JPL Horizons

# Create a class for the API
class API():
	"""
	A class which acts as the API to get the initial positions and velocities
	of the planetaries bodies

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

	# Initiate the object
	def __init__(self, year, month, day):
		""" Create the initial and final date used by the API using the given parameters

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

	# Finds the initial position of a body in the solar system from the sun
	def getPosition(self, nasa_id):
		""" Get the initial position of the planetary body using the API

		Parameters
		----------
		nasa_id : int
		    The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		
		Returns
		-------
		position : np.array()
			An array representing the initial position vector of the planetary body
		"""
		bodyData = Horizons(id=nasa_id, location="@sun", epochs={'start': self.date_i,'stop': self.date_f,'step': '1d'}).vectors().as_array()
		dataString = str(bodyData[0])
		vectors = dataString.split(", ")
		return np.array([float(vectors[3]), float(vectors[4]), float(vectors[5])])

	# Finds the initial velocity vector of a body in the solar system from the sun
	def getVelocity(self, nasa_id):
		""" Get the initial velocity vector of the planetary body using the API

		Parameters
		----------
		nasa_id : int
		    The ID used by NASA's JPL Horizons for each planet (from 1 to 8)
		
		Returns
		-------
		position : np.array()
			An array representing the initial velocity vector of the planetary body
		"""
		bodyData = Horizons(id=nasa_id, location="@sun", epochs={'start': self.date_i,'stop': self.date_f,'step': '1d'}).vectors().as_array()
		dataString = str(bodyData[0])
		vectors = dataString.split(", ")
		return np.array([float(vectors[6]), float(vectors[7]), float(vectors[8])])


# Create a class for planetaries bodies
class Body():
	"""
	A class which represents the planetary bodies

    Attributes
    ----------
    mass : float
		A float representing the planetary body's mass (kg)
	size : int
		An integer used for the marker size when drawing the planetary body
		in the graph
	color : str
		A string representing the color of the marker for the planetary body
		in the graph (using the xkcd color survey for the colors)
	name : str
		A string representing the name of the planetary body for easier indentification
	position : np.array()
		A numpy array representing the position vector of the planetary body
	velocity : np.array()
		A numpy array representing the velocity vector of the planetary body
	acceleration : np.array()
		A numpy array representing the acceleration vector of the planetary body
	force : np.array()
		A numpy array representing the force vector of the planetary body

    Methods
    -------
    addForce()
        Adds a gravitational force upon the planetary body
    resetForce()
    	Resets the planetary body's force to a zero vector
    move()
    	Calculates the body's velocity and position vectors using the Euler method
    draw()
    	Draws the planetary body onto the renderer's graph
    """

    # Initiate the object
	def __init__(self, mass, size, color, name):
		""" Create the initial required variables for a planetary body

		Parameters
		----------
		mass : float
			A float representing the planetary body's mass in kg
		size : int
			An integer used for the marker size when drawing the planetary body
			in the graph
		color : str
			A string representing the color of the marker for the planetary body
			in the graph (using the xkcd color survey for the colors)
		name : str
			A string representing the name of the planetary body for easier indentification
		"""
		self.mass = mass
		self.size = size
		self.color = color
		self.name = name
		self.position = np.zeros(3)
		self.velocity = np.zeros(3)
		self.acceleration = np.zeros(3)
		self.force = np.zeros(3)

	# Adds a force upon the body
	def addForce(self, addedForce):
		""" Add an additional gravitational force upon the planetary body

		Parameters
		----------
		addedForce : np.array()
			A numpy array representing the newly added force vector to the planetary body
		"""
		self.force = self.force + addedForce

	# Resets the force upon the body to zero
	def resetForce(self):
		""" Resets the body's force vector to a zero vector
		"""
		self.force = np.zeros(3)

	# 1 AU = 1.496e8 km = 1.496e11 m
	# Calculates the movement and changes the position of the body
	def move(self, dt):
		""" Calculate the body's velocity and position vectors using the Euler method

		Parameters
		----------
		dt : float
			A float representing the time steps used for the Euler method (s)
		"""
		# Transform from m/s and m to AU/d and AU
		self.acceleration = self.force / self.mass
		self.velocity += (self.acceleration * dt) * float(86400 / 1.496e11) # Float is conversion factor from m/s to AU/d
		self.position += self.velocity * (dt / float(86400))				# Float is conversion factor from s to days

    # Draws the planetary body on the 3D graph
	def draw(self, renderer):
		""" Draw the planetary body onto the renderer's graph

		Parameters
		----------
		renderer : Renderer()
			The renderer object used to host the figure and axes of the matplotlib graph
		"""
		renderer.ax.plot(*self.position, marker = "o", markersize = self.size, color = self.color)

# Create a class for a whole system
class System():
	"""
	A class which acts as the API to get the initial positions and velocities
	of the planetaries bodies

    Attributes
    ----------
    size : float
        A float representing the system's total radius size (AU)
    timeInterval : float
		A float representing the time steps that the system uses for physics (s)
	bodyArray : np.array()
		An array used to store all the planetary body objects included in the system

    Methods
    -------
    addBody()
        Adds a planetary body to the system's array
    calculateForce()
    	Calculates the gravitational forces between two planetary bodies
    update()
    	Calculates the forces, moves all the bodies, and draws all of them
    	for a single time step
    """

	# Initiate the object
	def __init__(self, size, timeInterval):
		""" Create the required variables for the system

		Parameters
		----------
		size : float
        	A float representing the system's total radius size (AU)
    	timeInterval : float
			A float representing the time steps that the system uses for physics (s)
		"""
		self.size = size
		self.timeInterval = timeInterval
		self.bodyArray = np.array([])

	# Adds a planetary body in the system's array list
	def addBody(self, body):
		""" Add a planetary body to the system's array

		Parameters
		----------
		body : Body()
        	A planetary body object that will be added to the system's body array
		"""
		self.bodyArray = np.append([self.bodyArray], body)

	# Calculates the gravitational force between two bodies in the system's array list
	def calculateForce(self, i, j):
		""" Calculate the graviational force between two planetary bodies using the
			formula F = (G * M1 * M2) / (r **2) and then converted into vectors

		Parameters
		----------
		i : int
        	An integer representing the index of the first planetary body in the system's
        	array of planetary bodies
    	j : int
			An integer representing the index of the second planetary body in the system's
        	array of planetary bodies
		"""
		# 1 AU = 1.496e8 km = 1.496e11 m
		vector = (self.bodyArray[j].position - self.bodyArray[i].position) * 1.496e11	# The number is to convert AU to m
		r = math.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))
		grav = (6.67e-11 * self.bodyArray[i].mass * self.bodyArray[j].mass) / (r ** 2)	# This is the force's magnitude
		self.bodyArray[i].addForce((grav / r) * vector)									# This converts the force's magnitude
		self.bodyArray[j].addForce(-(grav / r) * vector)								# to a force vector (N)

	# Updates each body 
	def update(self, renderer):
		""" Calculate the forces, moves all the planetary bodies, and draws all
			of them on the graphfor a single time step

		Parameters
		----------
		renderer : Renderer()
        	The renderer object used to host the figure and axes of the matplotlib graph
		"""
		# Calculates the gravitational forces between every body in the system's array list
		for i in range(0, self.bodyArray.size - 1):		# The two for loops allow to not repeat the
			for j in range(i + 1, self.bodyArray.size):	# same two body combinations twice
				self.calculateForce(i, j)
        # Moves and draws each planetary body in the system's array list
		for body in self.bodyArray:
			body.move(self.timeInterval)
			body.resetForce()
			body.draw(renderer)

# Create a class for the graphics renderer
class Renderer():
	"""
	A class which acts as the graphing renderer for the system and planetary bodies

    Attributes
    ----------
    framerate : int
        An integer representing the preferred framerate of the animation (fps)
    fig, ax : plt.subplots()
    	A subplot composed of a figure and axes to plot all the planetary bodies

    Methods
    -------
    resetGraph()
        Resets the graph's axes and does a basic animation using the pause and clear commands
    """

	# Initiate the object
	def __init__(self, framerate):
		""" Create the required variables for the renderer

		Parameters
		----------
		framerate : int
	        An integer representing the preferred framerate of the animation (fps)
    	fig, ax : plt.subplots()
    		A subplot composed of a figure and axes to plot all the planetary bodies
		"""
		self.framerate = framerate
		self.fig, self.ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
		self.fig.tight_layout() # Changes the layout to make it more readable

    # Resets the graphics to allow the next frame
	def resetGraph(self, size, getOrbit):
		""" Resets the graph's axes and does a basic animation using the pause and clear commands

		Parameters
		----------
		size : int
        	An integer representing the limits of the graph (AU)
        getOrbit : bool
        	A boolean used to know to either trace the orbit or not
		"""
		self.ax.set_xlim((-size, size))
		self.ax.set_ylim((-size, size))
		self.ax.set_zlim((-size, size))
		plt.pause(float(1/self.framerate))
		if getOrbit == False:	# Only clears all on the graph if orbits are not wanted
			self.ax.clear()

# Asks the user for the year, month, and day for the initial date of the simulation
year = int(input("What initial year does the simulation begin in?\n"))
month = int(input("What initial month does the simulation begin in?\n"))
day = int(input("What initial day does the simulation begin in?\n"))

# Creates the renderer and API
grapher = Renderer(60)
nasa = API(year, month, day)

# Sets arrays for the names, colors, and masses for each planetary body in the solar system
nameList = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])
colorList = np.array(["xkcd:goldenrod", "xkcd:grey", "xkcd:dark orange", "xkcd:dark blue", "xkcd:rust", "xkcd:light brown", "xkcd:tan", "xkcd:sea blue", "xkcd:grey blue"])
massList = np.array([1.989e30, 3.301e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.683e26, 8.681e25, 1.024e26])

# Sets an array for the maximum sizes of the system depending on how many planets are used
sizeList = np.array([0, 0.5, 1.0, 1.5, 2.0, 6.0, 10.0, 20.0, 30.0])

# Asks the use for the number of planets wanted and initiates the solar system with a time step of 1 day (86400s)
n = int(input("How many planets do you want?\n"))
solar = System(sizeList[n], float(86400*1))

# Adds every wanted planetary body to the solar system
for i in range(n+1):
	solar.addBody(Body(massList[i], 1, colorList[i], nameList[i]))
	print(f"Added planetary body {solar.bodyArray[i].name}!")

	# Sets the initial position and velocity vectors of each planetary body except for the sun
	if i != 0:
		solar.bodyArray[i].position = nasa.getPosition(i)
		print(f"The position of {solar.bodyArray[i].name} is {solar.bodyArray[i].position} AU.")
		solar.bodyArray[i].velocity = nasa.getVelocity(i)
		print(f"The velocity of {solar.bodyArray[i].name} is {solar.bodyArray[i].velocity} AU/d.")

# Renders n number of frames of the simulation
n = int(input('How many days do you want to simulate?\n'))

# Check if the user wishes to see the orbits
seeOrbit = input('Do you want to see the orbits (T/F)?\n')
if seeOrbit == 'T':
	seeOrbit = True
else:
	seeOrbit = False

# Checks if the user wishes to save the final frame of the graph
save = input('Do you want to save the result (T/F)?\n')
if save == 'T':		# Repeating the same if/else loops twice is not efficient
	save = True 	# and will be changed in the next version of the code
else:
	save = False

# Loops simulation for n days
for t in range(n):
	# Resets graph and then updates the graph to animate
	grapher.resetGraph(solar.size, seeOrbit)
	solar.update(grapher)

	# Prints the position of every planetary body and the frame number for debugging
	for body in solar.bodyArray:
		print(f'The position of {body.name} is {body.position} AU.')
	print(f'Frame {t+1} drawn!')	

# Saves the final frame of the graph if asked by the user
if save == True:
	plt.savefig('system.png', format='png', dpi=1200)	# The format can be changed between .png and .pdf
	print("Final frame of the graph is saved!")