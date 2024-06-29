"""
woodo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import CoolProp.CoolProp as cp
from scipy.integrate import cumtrapz

plt.style.use('bmh')

ELEVATION = 75  # degrees. Assumed constant
AREA = np.pi * (102e-3) ** 2 / 4

# Read the IMU data from file into a dataframe
imu_data = pd.read_csv("Processed data/data_norsk_radar_IMU.csv", header=0)
gps_data = pd.read_csv("Processed data/data_norsk_radar_GPS.csv", header=0)
balloon_data = pd.read_csv("../balloon/yep.csv", header=0)

# Y axis is along thrust direction
imu_time = imu_data['t_a_y_imu']
imu_a_y = imu_data['a_y_imu']


# Load engine data
engine_data = pd.read_csv("eng.csv", header=0)
engine_data['t'] = engine_data['t'] \
                 + (imu_time[np.argmax(imu_a_y)] - engine_data['t'].iloc[np.argmax(engine_data['f'])]) \
                 - 0.88
engine_data['m'] = 1e-3 * engine_data['m'] + 10.5 # Convert to kg and add dry mass


# Resampling all engine data to match the IMU sampling using interpolation
resampled_f = np.interp(imu_time, engine_data['t'], engine_data['f'])
resampled_m = np.interp(imu_time, engine_data['t'], engine_data['m'])

# TWR
TWR = resampled_f / (resampled_m * 9.81)

mag_y = imu_data['mag_y']
t_mag_y = imu_data['t_mag_y']



# FORCES in y direction
F_g = 9.81 * resampled_m * np.cos(np.radians(90-ELEVATION))
F_g = F_g * 0
F_net = 9.81 * resampled_m * imu_a_y
F_d = F_net - F_g - resampled_f

# Function for calculating the density at a given height from baloon data
def get_density(height: int) -> int:
    location = np.argmin(np.abs(balloon_data["height"] - height))

    pres = balloon_data["pressure"].iloc[location]
    temp = 273.15 + balloon_data["temperature"].iloc[location] 

    #Calculate density of the air using coolprop
    density = cp.PropsSI("D", "P", pres, "T", temp, "Air")

    return density

def get_mach(vel: int, height: int) -> int:
    location = np.argmin(np.abs(balloon_data["height"] - height))

    pres = balloon_data["pressure"].iloc[location]
    temp = 273.15 + balloon_data["temperature"].iloc[location] 

    #Calculate density of the air using coolprop
    speed_of_sound = cp.PropsSI("A", "P", pres, "T", temp, "Air")

    return vel / speed_of_sound

# Resampling the GPS height to match the IMU sampling
resampled_height = np.interp(imu_time, gps_data['t'], gps_data['height_smooth'])

density = np.asarray([get_density(height) for height in resampled_height])

# integrating force to get velocity using scipy
imu_vel_y = cumtrapz(9.81 * (imu_a_y - 1 * np.sin(np.radians(ELEVATION))) , imu_time, initial=0)

# Drag coefficient
C_d = -F_d / (density * imu_vel_y ** 2 * AREA / 2)
# C_d[imu_vel_y < 20] = 0
C_d = np.clip(C_d, 0, 10)

density_height = [get_density(height) for height in gps_data["height_smooth"]]

#plot mach number as fucntion of height
mach = [get_mach(vel, height) for vel, height in zip(imu_vel_y, resampled_height)]

plt.plot(imu_time, mach, label="Mach number")
# plt.twinx()
plt.plot(imu_time, imu_vel_y, label="Velocity", c="c")
plt.xlabel("Timestamp [s]")
plt.ylabel("Mach number")
plt.title("Mach number")
plt.legend()
plt.show()

# # Plot density as function of GPS height
# plt.plot(gps_data["height_smooth"], density_height, label="Air density")
# plt.xlabel("Height [m]")
# plt.ylabel("Density [kg/m^3]")
# plt.title("Air density as function of height")
# plt.legend()
# plt.show()


# Plot the IMU data
# plt.plot(imu_time, imu_a_y, label='IMU Acceleration Y')
# plt.plot(imu_time, TWR, label='TWR')
# plt.xlabel('Timestamp [s]')
# plt.ylabel('Acceleration [g]')
# plt.title('Accelerometer')
# plt.legend()
# plt.show()

# Plot the air density against height
plt.plot(gps_data['height_smooth'], density_height, label='Density')
plt.xlabel('Height [m]')
plt.ylabel('Density [kg/m^3]')
plt.title('Air Density')
plt.legend()
plt.show()

# plot gps height
plt.plot(gps_data['t'], gps_data['height_smooth'], label='GPS Height')
plt.xlabel('Timestamp [s]')
plt.ylabel('Height [m]')
plt.title('GPS Height')
plt.legend()
plt.show()

# Plot all forces
plt.plot(imu_time, F_net, label='Net Force', c="k")
plt.plot(imu_time, F_g, label='Gravity', c="g")
plt.plot(imu_time, F_d, label='Drag Force', c="b")
plt.plot(imu_time, resampled_f, label='Thrust', c="r")
plt.xlabel('Timestamp [s]')
plt.ylabel('Force [N]')
plt.title('Forces during flight')
plt.legend()
plt.show()

# Plot drag coefficient
plt.plot(imu_time, C_d, label='Drag Coefficient')
plt.xlabel('Timestamp [s]')
plt.ylabel('Drag Coefficient')
plt.title('Drag Coefficient during flight')
plt.legend()
# plt.ylim(0, 1)
plt.show()

# Plot the drag coefficient as a function of mach number
plt.plot(mach, C_d, label='Drag Coefficient')
plt.xlabel('Mach number')
plt.ylabel('Drag Coefficient')
plt.title('Drag Coefficient as a function of Mach number')
plt.legend()
# plt.ylim(0, 1)
plt.show()

#plot the imu magnetometer
# plt.plot(imu_data["t_mag_x"], imu_data['mag_x'], label='Magnetometer X')
plt.plot(imu_data["t_mag_y"], imu_data['mag_y'], label='Magnetometer Y')
# plt.plot(imu_data["t_mag_z"], imu_data['mag_z'], label='Magnetometer Z')
plt.xlabel('Timestamp [s]')
plt.ylabel('Magnetometer')
plt.title('Magnetometer')
plt.legend()
plt.show()

