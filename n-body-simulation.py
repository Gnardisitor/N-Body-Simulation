"""
N-Body Simulation

This script allows the user to simulate any section of the solar system and displays the final orbits, the initial and
final position of the planet in a final graph.

This version uses the Runge-Kutta fourth order method (RK4).

Author : Dragos Bajanica
Version : 2.0.0
"""

# Import required libraries
from math import sqrt                           # Used only for the function sqrt()
import numpy as np                              # Used for better arrays used as vectors
from tqdm import tqdm                           # Progress bar for terminal in python
import matplotlib.pyplot as plt                 # Used for graphing in three dimensions
from matplotlib.widgets import Button, Slider   # Used for buttons and sliders in a pyplot
from astroquery.jplhorizons import Horizons     # Used to get initial vectors from NASA's JPL Horizons


class Body:
    """
    The object for a planetary body in a system

    Attributes
    ----------
    mass : float
        The mass float of the planetary body
    color : str
        The string for the color markers of the body
    name : str
        The string of the body's given name
    body_data : Table
        The table of data about the planetary body
    position : np.ndarray
        The position vector array of the body
    velocity : np.ndarray
        The velocity vector array of the body
    acceleration : np.ndarray
        The acceleration vector array of the body
    history_x : np.ndarray
        The array of x positions of the body
    history_y : np.ndarray
        The array of y positions of the body
    history_z : np.ndarray
        The array of z positions of the body

    Methods
    -------
    get_acceleration()
        Returns the acceleration vector array of the planetary body
    """

    # Define used variables for the class
    __slots__ = ("mass", "color", "name", "body_data", "position", "velocity", "acceleration", "history_x", "history_y",
                 "history_z")

    # Define constants for calculations
    DISTANCE_CONVERSION = 1.496e11
    TIME_CONVERSION = 86400.0
    ACCELERATION_CONVERSION = (TIME_CONVERSION ** 2) / DISTANCE_CONVERSION
    G = 6.67e-11

    # Sets arrays for the names, colors, and masses for each planetary body in the solar system
    NAMES = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])
    COLORS = np.array(["xkcd:goldenrod", "xkcd:grey", "xkcd:dark orange", "xkcd:clear blue", "xkcd:rust",
                       "xkcd:light brown", "xkcd:tan", "xkcd:sea blue", "xkcd:grey blue"])
    MASSES = np.array([1.989e30, 3.301e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.683e26, 8.681e25, 1.024e26])

    def __init__(self, nasa_id: int, year: int):
        """
        Initializes the planetary body object

        Parameters
        ----------
        nasa_id : int
            An integer of the NASA id of the planetary body
        year : int
            An integer for the beginning year of the simulation
        """

        # Sets the qualitative properties of the body from the predefined arrays from the id
        self.mass = self.MASSES[nasa_id]
        self.color = self.COLORS[nasa_id]
        self.name = self.NAMES[nasa_id]

        # Use JPL Horizons to initialize the position and velocity vectors of the bodies
        self.body_data = Horizons(id=nasa_id, location="@sun",
                                  epochs={'start': f'{year}-01-01', 'stop': f'{year}-01-02', 'step': '1d'}).vectors()

        if nasa_id != 0:
            self.position = np.array([self.body_data["x"][0], self.body_data["y"][0], self.body_data["z"][0]])
        else:
            self.position = np.zeros(3)
        self.velocity = np.array([self.body_data["vx"][0], self.body_data["vy"][0], self.body_data["vz"][0]])
        self.acceleration = np.zeros(3)

        # Create an array used to store all the body's position history for the simulation
        self.history_x = np.array([])
        self.history_y = np.array([])
        self.history_z = np.array([])

    def get_acceleration(self, body: "Body"):
        """
        Returns the acceleration vector array of the planetary body

        Parameters
        ----------
        body : Body
            The other body object used to compute acceleration

        Returns
        -------
        np.ndarray
            The acceleration vector array from gravity between both bodies
        """

        vector = (body.position - self.position) * self.DISTANCE_CONVERSION     # Measured in m (vector)
        r = sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))        # Measured in m (scalar)
        acc = ((self.G * body.mass * vector) / (r ** 3))                        # Measured in m/s^2
        return acc * self.ACCELERATION_CONVERSION                               # Measured in AU/d^2


class System:
    """
    The object for a system composed of planetary bodies

    Attributes
    ----------
    n : int
        An integer of the number of bodies in the system
    h : float
        A float of the time interval for the simulation
    year : int
        An integer of the year of the beginning of the simulation
    bodies : np.ndarray
        An array of all the planetary bodies in the system

    Methods
    -------
    set_acceleration()
        Sets the acceleration of a body with all other bodies in the system
    f()
        Computes the derivative of the position and velocity vectors given
    rk4()
        Computes the new position and velocity using Runge-Kutta Fourth Order method
    update()
        Updates the simulation for one time interval h
    """

    # Define used variables for the class
    __slots__ = ("n", "h", "year", "bodies")

    def __init__(self, n: int, h: float, year: int):
        """
        Initializes the system object

        Parameters
        ----------
        n : int
            An integer of the number of bodies in the system
        h : float
            A float of the time interval for the simulation
        year : int
            An integer of the year of the beginning of the simulation
        """

        # Set time interval and number of planets for the simulation
        self.n = n
        self.h = h
        self.year = year

        # Add all the planetary bodies to the system's array
        self.bodies = np.array([])
        for i in range(self.n + 1):
            self.bodies = np.append(self.bodies, Body(i, self.year))

        # Remove any additional velocity from the center of mass
        temp_vel = np.zeros(3)
        temp_mass = 0.0
        for body in self.bodies:
            temp_vel = temp_vel + (body.mass * body.velocity)
            temp_mass = temp_mass + body.mass
        temp_vel = temp_vel / temp_mass
        for body in self.bodies:
            body.velocity = body.velocity - temp_vel

    def set_acceleration(self, nasa_id: int):
        """
        Sets the acceleration of a body with all other bodies in the system

        Parameters
        ----------
        nasa_id : int
            An integer of the NASA id of the planetary body
        """

        # Create temporary variable
        temp_acc = np.zeros(3)

        # Get all the accelerations from all the planets
        for body in self.bodies:
            if body.mass != self.bodies[nasa_id].mass:
                temp_acc = temp_acc + self.bodies[nasa_id].get_acceleration(body)

        # Change acceleration value
        self.bodies[nasa_id].acceleration = temp_acc

    def f(self, y: np.ndarray, k: np.ndarray, nasa_id: int):
        """
        Computes the derivative of the position and velocity vectors given

        Parameters
        ----------
        y : np.ndarray
            The initial derivative of the position and velocity vectors
        k : np.ndarray
            The new vectors at the specified position
        nasa_id : int
            An integer of the NASA id of the planetary body

        Returns
        -------
        np.ndarray
            The derivative of the position and velocity vectors
        """

        # Save initial position
        init_pos = self.bodies[nasa_id].position

        # Replace with new position
        self.bodies[nasa_id].position = init_pos + k[0]

        # Compute the new
        self.set_acceleration(nasa_id)
        new_vel = y[0] + k[1]
        new_acc = self.bodies[nasa_id].acceleration

        # Reset vectors and return values
        self.bodies[nasa_id].position = init_pos
        self.bodies[nasa_id].velocity = y[0]
        self.bodies[nasa_id].acceleration = y[1]
        return np.array([new_vel, new_acc])

    def rk4(self, nasa_id: int):
        """
        Computes the new position and velocity using Runge-Kutta Fourth Order method

        Parameters
        ----------
        nasa_id : int
            An integer of the NASA id of the planetary body
        """

        # Set y as velocity and acceleration vectors (derivatives of position and velocity)
        self.set_acceleration(nasa_id)
        y = np.array([self.bodies[nasa_id].velocity, self.bodies[nasa_id].acceleration])

        # Calculate 4 different Euler methods at different times
        k1 = y  # k1 is vel & acc
        k2 = self.f(y, self.h * (k1 / 2), nasa_id)  # Finds k2 using k1
        k3 = self.f(y, self.h * (k2 / 2), nasa_id)  # Finds k3 using k2
        k4 = self.f(y, self.h * k3, nasa_id)  # Finds k4 using k3

        # Do the weighted average of k1, k2, k3, k4
        self.bodies[nasa_id].position = self.bodies[nasa_id].position + (
                    (self.h / 6) * (k1[0] + (2 * k2[0]) + (2 * k3[0]) + k4[0]))
        self.bodies[nasa_id].velocity = self.bodies[nasa_id].velocity + (
                    (self.h / 6) * (k1[1] + (2 * k2[1]) + (2 * k3[1]) + k4[1]))

        # Add new position vector to the history arrays
        self.bodies[nasa_id].history_x = np.append(self.bodies[nasa_id].history_x, self.bodies[nasa_id].position[0])
        self.bodies[nasa_id].history_y = np.append(self.bodies[nasa_id].history_y, self.bodies[nasa_id].position[1])
        self.bodies[nasa_id].history_z = np.append(self.bodies[nasa_id].history_z, self.bodies[nasa_id].position[2])

    def update(self):
        """
        Updates the simulation for one time interval h
        """

        # Computes RK4 for every planetary body
        for nasa_id in range(self.n + 1):
            self.rk4(nasa_id)


# The constant array of sizes for the graph
SIZES = np.array([0, 0.5, 1.0, 1.5, 2.0, 6.0, 10.0, 20.0, 30.0])

# The constant of the time interval
H = 1.0

# Create the graph
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

# Set the positions of the sliders and buttons of the graph
fig.subplots_adjust(bottom=0.25, left=0.15)
ax_body = fig.add_axes([0.25, 0.15, 0.65, 0.03])
step_body = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
ax_year = fig.add_axes([0.25, 0.1, 0.65, 0.03])
step_year = np.array(range(1000, 2025))
ax_days = fig.add_axes([0.15, 0.25, 0.03, 0.65])
step_days = np.array(range(0, 50001, 100))
ax_set = fig.add_axes([0.25, 0.05, 0.10, 0.03])
ax_run = fig.add_axes([0.40, 0.05, 0.10, 0.03])

# Create the sliders and buttons of the graph
body_slider = Slider(ax=ax_body, label='Planets', valmin=1, valstep=step_body, valmax=8, valinit=4,
                     color="xkcd:clear blue")
year_slider = Slider(ax=ax_year, label='Year', valmin=1000, valstep=step_year, valmax=2024, valinit=2000,
                     color="xkcd:clear blue")
days_slider = Slider(ax=ax_days, label='Iterations', valmin=0, valstep=step_days, valmax=50000, valinit=5000,
                     color="xkcd:clear blue", orientation="vertical")
set_button = Button(ax_set, 'Set', hovercolor="xkcd:clear blue")
run_button = Button(ax_run, 'Run', hovercolor="xkcd:clear blue")

# Start the graph
plt.ion()
plt.show()
plt.pause(0.001)

# Initial system with default values
year = int(year_slider.val)
n = int(body_slider.val)
solar = System(n, H, year)
ax.set_xlim((-SIZES[n], SIZES[n]))
ax.set_ylim((-SIZES[n], SIZES[n]))
ax.set_zlim((-SIZES[n], SIZES[n]))


# Update the system with new values
def set_system(event):
    global solar
    new_year = int(year_slider.val)
    new_n = int(body_slider.val)
    solar = System(new_n, H, new_year)
    ax.set_xlim((-SIZES[new_n], SIZES[new_n]))
    ax.set_ylim((-SIZES[new_n], SIZES[new_n]))
    ax.set_zlim((-SIZES[new_n], SIZES[new_n]))


can_run = False


# Allow to run the simulation
def run(event):
    global can_run
    can_run = True


# Wait until run is clicked
while can_run is False:
    set_button.on_clicked(set_system)
    run_button.on_clicked(run)
    plt.pause(0.1)

# Run the simulation for the number of iterations
days = int(days_slider.val)
bar = tqdm(total=days)
for t in np.arange(0, float(days), H):
    solar.update()
    bar.update(H)
bar.close()

# Plot the orbits of the planetary bodies on the graph
for body in solar.bodies:
    ax.plot(*np.array([body.history_x[::10], body.history_y[::10], body.history_z[::10]]),
            marker=".", markersize=1, alpha=0.2, color=body.color)
    ax.plot(*np.array([body.history_x[0], body.history_y[0], body.history_z[0]]),
            marker="o", markersize=2, alpha=0.5, color=body.color)
    ax.plot(*np.array([body.history_x[-1], body.history_y[-1], body.history_z[-1]]),
            marker="o", markersize=3, alpha=1.0, color=body.color)

# Show the orbits
plt.show(block=True)
