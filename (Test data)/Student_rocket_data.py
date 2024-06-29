# -*- coding: utf-8 -*-
'''
This script refines raw Dewesoft data from an Education student rocket. The
different data streams are separated and converted to physical units, stored,
smoothed, processed and presented graphically for preliminary analysis.

The code presented below is meant to serve as a starting point for customized
data analysis, and should be expanded upon and edited as needed.

Andøya Space Education

Created on Tue Aug 3 2021 at 16:20:00.
Last modified [dd.mm.yyyy]: 06.06.2024
@author: bjarne.ådnanes.bergtun
The tkinter code is based on a script by odd-einar.cedervall.nervik
'''

import tkinter as tk # GUI
import tkinter.filedialog as fd # file dialogs
import os # OS-specific directory manipulation
from os import path # common file path manipulations
import dask.dataframe as dd # import of data
import matplotlib.pyplot as plt # plotting
import numpy as np # maths
import pandas as pd # data handling


# Setup of imported libraries

pd.options.mode.chained_assignment = None


# =========================================================================== #
# ========================= User defined parameters ========================= #

# Logical switches.
# If using Spyder, you can avoid needing to load the data every time by going
# to "Run > Configure per file" and activating the option "Run in console's
# namespace instead of an empty one".

load_data = True
sanitize_gps = True # Filter GPS-data using satellite number & height?
convert_data = True
process_data = True # Relevant calculations for your scientific case
create_plots = True
show_plots = True
export_plots = True # Plots cannot be exported unless created!
export_processed_data = True # Simplified channel names
export_raw_data = False # Original channel names
export_kml = False


# To save memory and computing power, this script allows users to exclude all
# data before a given time t_0.

t_0 = 0 # [s]
t_end = np.inf # [s]


# ============================ Sensor parameters ============================ #

# analogue accelerometers
a_x_sens = -0.045 # Sensitivity [V/gee]
a_x_offset = 2.529 # Offset [V]. Nominal value: 2.5 V
a_x_max = 50 # Sensor limit in the x-direction [gee]

a_y_sens = 0.125 # Sensitivity [V/gee]
a_y_offset = 2.424 # Offset [V]. Nominal value: 2.5 V
a_y_max = 20 # Sensor limit in the y-direction [gee]


# External and internal temperature sensors
temp_ext_gain = 9.8 # Amplification of the sensor output [-].
temp_ext_sens = 10. # Sensitivity of the sensor itself, sans gain [mV/K]
temp_ext_offset = 9.8*0.5 # Sensor output at 0 degrees celsius (before gain) [V]
temp_int_gain = 6.72 # Amplification of the sensor output [-].
temp_int_sens = 10. # Sensitivity of the sensor itself, sans gain [mV/K]
temp_int_offset = 6.72*0.5 # Sensor output at 0 degrees celsius (before gain) [V]


# NTC
R_fixed = 1E4 # [ohm]
R_ref = 1E4 # [ohm]
A_1 = 3.354016E-3 # [1/K]
B_1 = 2.569850E-4 # [1/K]
C_1 = 2.620131E-6 # [1/K]
D_1 = 6.383091E-8 # [1/K]


# IMU
a_x_imu_sens = 7.32E-4 # Sensitivity [gee/LSB]
a_y_imu_sens = 7.32E-4 # Sensitivity [gee/LSB]
a_z_imu_sens = 7.32E-4 # Sensitivity [gee/LSB]
a_x_imu_offset = 0 # Output at 0 gee [signed integer bit value]
a_y_imu_offset = 0 # Output at 0 gee [signed integer bit value]
a_z_imu_offset = 0 # Output at 0 gee [signed integer bit value]

ang_vel_x_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_y_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_z_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_x_offset = 0 # Output at 0 dps [signed integer bit value]
ang_vel_y_offset = 0 # Output at 0 dps [signed integer bit value]
ang_vel_z_offset = 0 # Output at 0 dps [signed integer bit value]

mag_x_sens = 1.4E-4 # Sensitivity [gauss/LSB]
mag_y_sens = 1.4E-4 # Sensitivity [gauss/LSB]
mag_z_sens = 1.4E-4 # Sensitivity [gauss/LSB]
mag_x_offset = 0 # Output at 0 gauss [signed integer bit value]
mag_y_offset = 0 # Output at 0 gauss [signed integer bit value]
mag_z_offset = 0 # Output at 0 gauss [signed integer bit value]


# Power sensor
voltage_sensor_gain = 0.288 # As a fraction, not in dB!
R_current_sensor = 20E-3 # [ohm]


# Payload
a7_occupied = False # [bool]

# ============================= Channel set-up ============================== #

# The dictionaries below serves several purposes:
#
#    1) Limit which channels are loaded (to ensure greater computational
#       performance)
#
#    2) Identify the data streams with the correct sensors
#
#    3) Simplify and/or clarify channel names. Among other things, DeweSoft
#       appends the data unit to the channel names (i.e. 'Time (s)' rather
#       than 'Time'). Some channel names are also rather cryptic
#
# It is cruical that the lefmost channel names corresponds exactly to the
# column names used in the raw data file!!!

# ====================== Channel names below this line ====================== #

analogue_channels = {
    'P0/Main stream/pressure ()': 'pressure',
    'P0/Main stream/optical ()': 'light',
    'P0/Main stream/battery_voltage ()': 'a7',
    'P0/Main stream/external_temperature ()': 'temp_ext',
    'P0/Main stream/internal_temperature ()': 'temp_int',
    'P0/Main stream/accelerometer_x ()': 'a_x',
    'P0/Main stream/accelerometer_y ()': 'a_y',
    'P0/Main stream/magnetometer ()': 'mag',
    }

temp_array_channels = {
    'P0/Temp array/a_t0 ()': 'temp_array_0',
    'P0/Temp array/a_t1 ()': 'temp_array_1',
    'P0/Temp array/a_t2 ()': 'temp_array_2',
    'P0/Temp array/a_t3 ()': 'temp_array_3',
    'P0/Temp array/a_t4 ()': 'temp_array_4',
    'P0/Temp array/a_t5 ()': 'temp_array_5',
    'P0/Temp array/a_t6 ()': 'temp_array_6',
    'P0/Temp array/a_t7 ()': 'temp_array_7',
    'P0/Temp array/a_t8 ()': 'temp_array_8',
    'P0/Temp array/a_t9 ()': 'temp_array_9',
    }

power_sensor_channels = {
    'P0/Temp array/array_voltage ()': 'voltage',
    'P0/Temp array/array_current ()': 'current',
    }

gps_channels = {
    'P0/GPS/IMU/GPS_satellites ()': 'satellites',
    'P0/GPS/IMU/GPS_longitude ()': 'long',
    'P0/GPS/IMU/GPS_latitude ()': 'lat',
    'P0/GPS/IMU/GPS_altitude ()': 'height',
    'P0/GPS/IMU/GPS_speed ()': 'speed',
    'P0/GPS/IMU/GPS_GDOP ()': 'gdop',
    }

imu_channels = {
    'P0/GPS/IMU/IMU_Ax ()': 'a_x_imu',
    'P0/GPS/IMU/IMU_Ay ()': 'a_y_imu',
    'P0/GPS/IMU/IMU_Az ()': 'a_z_imu',
    'P0/GPS/IMU/IMU_Gx ()': 'ang_vel_x',
    'P0/GPS/IMU/IMU_Gy ()': 'ang_vel_y',
    'P0/GPS/IMU/IMU_Gz ()': 'ang_vel_z',
    'P0/GPS/IMU/IMU_Mx ()': 'mag_x',
    'P0/GPS/IMU/IMU_My ()': 'mag_y',
    'P0/GPS/IMU/IMU_Mz ()': 'mag_z',
    }

misc_channels = {
    'Time (s)': 't',
    'P0/Main stream/main_counter ()': 'framecounter',
    }

# ======================= Don't copy over this line! ======================== #

# Create one large dictionary of all channels to be imported

channels = {
    **analogue_channels,
    **temp_array_channels,
    **power_sensor_channels,
    **gps_channels,
    **imu_channels,
    **misc_channels
    }


# Replace dictionaries with list of new channel names. This makes it easy to
# write code for, say, all the temp-array sensors, as the entire list can be
# reached by using "temp_array_channels" or, alternatively
# "x in temp_array_channels". See further down in the code for examples.

analogue_channels = list(analogue_channels.values())
temp_array_channels = list(temp_array_channels.values())
power_sensor_channels = list(power_sensor_channels.values())
gps_channels = list(gps_channels.values())
imu_channels = list(imu_channels.values())
misc_channels = list(misc_channels.values())



# =========================================================================== #
# ============================== Load CSV data ============================== #

if load_data:
    
    # Pro tip: If variables are being saved between script runs, all file
    # dialogs can be suppressed by commenting out the next two lines. If
    # variables are not being saved, one can instead manually copy and paste
    # the wanted file paths (replace '').
    data_file = ''
    working_directory = ''
    
    
    if data_file == '':
        # First a root window is created and put on top of all other windows.
    
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
    
        # On top of the root window, a filedialog is opened to get the CSV file
        # from the file explorer.
        data_file = fd.askopenfilename(
            title = 'Select rocket data to import',
            filetypes = (('CSV files','.csv'),('All files','.*')),
            parent = root,
            )
    
    print('\nLoading data from t =',t_0,'s to',t_end,'s.')
    print('\nFile path:\n' + data_file + '\n')

    # Save some file paths for later.

    parent_file_name, parent_file_extension = path.splitext(
        path.basename(data_file)
        )

    # Use dask to load the file, saving only the lines with t >= t_0 into
    # memory, and only the colons listed in the channels dictionary defined
    # above.

    raw_data = dd.read_csv(
        data_file,
        usecols = channels.keys(),
        sep = ',',
        assume_missing = True,
        encoding = 'windows-1252', # 'utf-8',
        )
    raw_data = raw_data[raw_data['Time (s)'] >= t_0]
    raw_data = raw_data[raw_data['Time (s)'] < t_end]
    raw_data = raw_data.compute()

    # Simplify channel names according to the user-defined dictionary.

    raw_data.rename(
        columns = channels,
        inplace = True,
        )
    
    
    if not 'speed' in gps_channels: # Check if using old or new GPS setup.
        raw_data['speed'] = np.sqrt(raw_data['v_x']**2 + raw_data['v_y']**2 + raw_data['v_z']**2)
        gps_channels += ['speed']


    # Sanitize GPS-data
    if sanitize_gps:
        mask = raw_data['satellites'] < 3
        mask2 = raw_data['satellites'] == 3
        mask3 = raw_data['speed'] > 1e6 # cm/s
        mask4 = raw_data['height'] > 1e6 # cm
        raw_data.loc[mask, ['lat', 'long', 'height', 'speed']] = np.nan
        raw_data.loc[mask2, ['height', 'speed']] = np.nan
        raw_data.loc[mask3, ['speed']] = np.nan
        raw_data.loc[mask4, ['height']] = np.nan
    
    
    # ======================== De-multiplex data ======================== #
    
    def isolate_dataseries(data_label, unique_time_label=False):
        time_label = 't'
        if unique_time_label:
            time_label += '_' + data_label
        data = pd.DataFrame()
        data[[time_label, data_label]] = raw_data[['t', data_label]].dropna(thresh=2)
        # data = data.dropna(subset=[time_label]) # In case we somehow have a NaN in the time-axis
        # To ease the identification of lost data, we want to store a nan-value
        # between data points where there *should* have been data. In order to
        # identify these points, we need to calculate how often the sensor data
        # should come. The code below is an automated solution; a less error-prone
        # but more manual method would be to calculate the expected measurement
        # frequency from the frame format specification
        dt = min(data[time_label].dropna().diff())
        # No time measurement is exact, but if the interval between each data point
        # is more than 1.5 the minimal update interval, we have missing data
        mask = np.abs(data[time_label].diff() - dt) / dt >= 0.5
        nan_data = data[mask]
        nan_data[time_label] -= dt
        nan_data.index -= 1
        nan_column = np.full(len(nan_data[data_label]), np.nan)
        nan_data[data_label] = nan_column
        data = pd.concat([data, nan_data]).sort_values(by=[time_label])
        data.reset_index(drop=True, inplace=True)
        return data
    
    def isolate_dataframe(frame, unique_time_label=False):
        # Step 1: Create list of separated data series
        # Step 2: Combine the data series into one dataframe
        # Step 3: Profit!
        if raw_data[frame].dropna(thresh=1).empty:
            frame += ['t']
            return raw_data[frame].dropna(thresh=2)
        if unique_time_label:
            dataseries = [isolate_dataseries(x, unique_time_label=True) for x in frame]
            new_frame = pd.concat(dataseries, axis=1, join='inner')
        else:
            new_frame = pd.DataFrame({'t':[]}) # Empty time-frame
            for x in frame:
                if x != 't':
                    dataseries = isolate_dataseries(x)
                    new_frame = pd.merge(new_frame, dataseries, how='outer')
            new_frame = new_frame.sort_values('t', ignore_index=True)
        return new_frame
    
    print('De-multiplexing ...')
    
    analogue = isolate_dataframe(analogue_channels)
    temp_array = isolate_dataframe(temp_array_channels)
    power_sensor = isolate_dataframe(power_sensor_channels)
    gps = isolate_dataframe(gps_channels)
    imu = isolate_dataframe(imu_channels, unique_time_label=True)
    misc = isolate_dataframe(misc_channels)
    
    
    # Create raw dataframes in order to enable re-runs of the conversion
    # section of the script without having to re-load data.
    
    analogue_raw = analogue.copy()
    power_sensor_raw = power_sensor.copy()
    temp_array_raw = temp_array.copy()
    gps_raw = gps.copy()
    imu_raw = imu.copy()
    

# =========================================================================== #
# ============================== Convert data =============================== #

if convert_data:

    print('Converting data to sensible units ...')

    # Physical constants

    T_0 = 273.15 # 0 celsius degrees in kelvin


    # Other useful constants
    U_main = 5.0
    U_array = 3.3
    wordlength_main = 8
    wordlength_array = 12


    # Conversion formulas

    def volt(bit_value, wordlength=wordlength_main, U=U_main):
        Z = 2**wordlength - 1
        rel_value = bit_value/Z
        rel_value[rel_value>1] = np.nan # Removes obious erronous data.
        return U*rel_value
    
    def analogue_voltage(U): # unit: volts
        return U*4.2 # Inverse gain from the encoder documentation

    def volt_to_pressure(U): # unit: kPa
        return (200*U+95)/9

    def linear_temp(U, gain, sensitivity, offset): # unit: celsius degrees
        return 1000*(U - offset)/(gain*sensitivity)

    def volt_to_acceleration(U, sensitivity, offset): # unit: gee
        return (U-offset)/sensitivity

    def phototransistor(U):
        # Not implemented!
        return U

    def magnetometer(U):
        # Not implemented!
        return U

    def NTC(bit_value, R_fix=R_fixed): # unit: celsius degrees
        divisor = (2**wordlength_array - 1 - bit_value) * R_ref
        divisor[divisor<=0] = np.nan # avoids division by zero
        R = R_fix * bit_value / divisor
        R[R<=0] = np.nan # avoids complex logarithms
        ln_R = np.log(R)
        T = 1/(A_1 + B_1*ln_R + C_1*ln_R**2 + D_1*ln_R**3)
        T -= T_0 # convert to celsius degrees
        return T

    def array_voltage(U, gain): # unit: volts
        return U/gain

    def array_current(U, R_current): # unit: ampere
        return U/(100*R_current)

    def imu_1D(bit_value, sensitivity, offset):
        """
        Parameters
        ----------
        bit_value : int
            Bit value to be converted.

        sensitivity : float
            Sensitivity of the sensor. Typical values can be found in the data sheet, but accurate values need to be determined experimentally.

        offset : signed int, optional
            Offset as a signed integer bit valye. The default is zero. Should be set to whatever signed integer one receives in DEWESoft when the measurment *should* be zero.

        Returns
        -------
        float
            Converted value(s). The unit depends on the sensitivity. Using the standard values gives the following units:
                accelerometer: gee, i.e. 9.81 m/s^2
                gyroscope: degrees per second
                magnetometer: gauss

        """
        return sensitivity*(bit_value-offset)

    def gps_degrees(angle): # unit: degrees
        return angle*1e-7

    def gps_height(height): # unit: meters
        return height*1e-2

    def gps_velocity(velocity): # unit: meters per second
        return velocity*1e-2
    
    
    # Convert raw channels to volt
    
    for x in analogue_channels:
        analogue[x] = volt(analogue_raw[x])
    
    for x in power_sensor_channels:
        power_sensor[x] = volt(
            power_sensor_raw[x],
            wordlength = wordlength_array,
            U = U_array,
            )
    
    
    # Create temporary dataframes in order to enable re-runs of the conversion
    # section of the script without having to re-load data.
    
    analogue_volt = analogue.copy()
    power_sensor_volt = power_sensor.copy()


    # Convert data to physical units

    analogue['pressure'] = volt_to_pressure(analogue_volt['pressure'])
    analogue['a_x'] = volt_to_acceleration(
        analogue_volt['a_x'],
        a_x_sens,
        a_x_offset,
        )
    analogue['a_y'] = volt_to_acceleration(
        analogue_volt['a_y'],
        a_y_sens,
        a_y_offset,
        )
    analogue['temp_int'] = linear_temp(
        analogue_volt['temp_int'],
        temp_int_gain,
        temp_int_sens,
        temp_int_offset,
        )
    analogue['temp_ext'] = linear_temp(
        analogue_volt['temp_ext'],
        temp_ext_gain,
        temp_ext_sens,
        temp_ext_offset,
        )
    analogue['light'] = phototransistor(analogue_volt['light'])
    analogue['mag'] = magnetometer(analogue_volt['mag'])
    
    if a7_occupied:
        analogue['a7'] = analogue_volt['a7']
    else:
        analogue['a7'] = analogue_voltage(analogue_volt['a7'])
   
    
    power_sensor['voltage'] = array_voltage(
        power_sensor_volt['voltage'],
        voltage_sensor_gain,
        )
    power_sensor['current'] = array_current(
        power_sensor_volt['current'],
        R_current_sensor,
        )


    gps['lat'] = gps_degrees(gps_raw['lat'])
    gps['long'] = gps_degrees(gps_raw['long'])
    gps['height'] = gps_height(gps_raw['height'])
    gps['speed'] = gps_velocity(gps_raw['speed'])


    imu['a_x_imu'] = imu_1D(imu_raw['a_x_imu'], a_x_imu_sens, a_x_imu_offset)
    imu['a_y_imu'] = imu_1D(imu_raw['a_y_imu'], a_y_imu_sens, a_y_imu_offset)
    imu['a_z_imu'] = imu_1D(imu_raw['a_z_imu'], a_z_imu_sens, a_z_imu_offset)

    imu['ang_vel_x'] = imu_1D(imu_raw['ang_vel_x'], ang_vel_x_sens, ang_vel_x_offset)
    imu['ang_vel_y'] = imu_1D(imu_raw['ang_vel_y'], ang_vel_y_sens, ang_vel_y_offset)
    imu['ang_vel_z'] = imu_1D(imu_raw['ang_vel_z'], ang_vel_z_sens, ang_vel_z_offset)

    imu['mag_x'] = imu_1D(imu_raw['mag_x'], mag_x_sens, mag_x_offset)
    imu['mag_y'] = imu_1D(imu_raw['mag_y'], mag_y_sens, mag_y_offset)
    imu['mag_z'] = imu_1D(imu_raw['mag_z'], mag_z_sens, mag_z_offset)


    for x in temp_array_channels:
        temp_array[x] = NTC(temp_array_raw[x])
    
    
    # Delete temporary dataframes
    del analogue_volt
    del power_sensor_volt


# =========================================================================== #
# ============================= Processes data ============================== #

if process_data:

    print('Calculating useful stuff ...')

    # smoothing function

    def smooth(data, r_tol=0.02, dense=True):
        """
        Smooths a data series by requiring that the change between one datapoint and the next is below a certain relative treshold (the default is 2 % of the total range of the values in the dataseries). If this is not the case, the value is replaced by NaN. This gives a 'rougher' output than a more sophisticated smoothing algorithm (see for example statsmodels.nonparametric.filterers_lowess.lowess from the statmodels module), but has the advantage of being very quick. If more sophisticated methods are needed, this algorithm can be used to trow out obviously erroneous data before the data is sent through a more traditional smoothing algorithm.

        Parameters
        ----------
        data : pandas.DataSeries
            The data series to be smoothed

        r_tol : float, optional
            Tolerated change between one datapoint and the next, relative to the full range of values in DATA. The default is 0.02.
            
        dense : bool, optional
            Whether the returned series should be dense (i.e. devoid of nan-values) or not.

        Returns
        -------
        data_smooth : pandas.DataSeries
            smoothed data series.

        """
        valid_data = data[data.notna()]
        if len(valid_data) == 0:
            data_range = np.inf
        else:
            data_range = np.ptp(valid_data)
        tol = r_tol*data_range
        data_smooth = data.copy()
        data_interpol = data_smooth.interpolate()
        data_smooth[np.abs(data_interpol.diff()) > tol] = np.nan
        if dense:
            data_smooth = data_smooth.interpolate()
        return data_smooth

    # Calculate smoothed data channels

    analogue['pressure_smooth'] = smooth(analogue['pressure'])
    analogue['mag_smooth'] = smooth(analogue['mag'])
    analogue['a_x_smooth'] = smooth(analogue['a_x'])
    analogue['a_y_smooth'] = smooth(analogue['a_y'])
    analogue['temp_int_smooth'] = smooth(analogue['temp_int'])
    analogue['temp_ext_smooth'] = smooth(analogue['temp_ext'])
    
    for x in temp_array_channels:
        temp_array[x+'_smooth'] = smooth(temp_array[x], r_tol=1e-6)
    temp_array_channels_smooth = [n + '_smooth' for n in temp_array_channels]

    gps['lat_smooth'] = smooth(gps['lat'], r_tol=.001)
    gps['long_smooth'] = smooth(gps['long'], r_tol=.001)
    gps['height_smooth'] = smooth(gps['height'], r_tol=.001)
    gps['speed_smooth'] = smooth(gps['speed'], dense=False)
    
    
    # ================ Calculations for scientific case ================= #




# =========================================================================== #
# =========================== Prepare for export ============================ #

export = export_processed_data or export_raw_data or export_kml or export_plots

if export:
    if working_directory == '':
        working_directory = fd.askdirectory(
            title = 'Choose output folder',
            parent = root
            )
    processed_data_directory = path.join(working_directory, 'Processed data')
    plot_directory = path.join(working_directory, 'Plots')


# =========================================================================== #
# ================================ Plot data ================================ #

if create_plots:

    print('Plotting ...')

    plt.ioff() # Prevent figures from showing unless calling plt.show()
    # plt.style.use('seaborn-darkgrid') # plotting style.
    plt.rcParams['legend.frameon'] = 'True' # Fill the background of legends.

    if export_plots and not path.exists(plot_directory):
        os.mkdir(plot_directory)
    
    # ==================== Custom plotting functions ==================== #

    # Custom parameters

    standard_linewidth = 0.5


    # First some auxillary functions containing some often-needed lines of
    # code for custom plots.


    # Standard settings for plt.figure()
    # Create a figure, or ready an already existing figure for new data.
    # Returns the window title for easier export with finalize_figure().

    def create_figure(name):
        name = name + ' [' + parent_file_name + ']'
        plt.figure(name, clear=True)
        return name


    # Standard plotting function.

    def plot_data(x, y, data=analogue, data_set=''):
        x_smooth = x + '_smooth'
        y_smooth = y + '_smooth'
        if x_smooth in data.columns:
            if data_set=='':
                data_set = 'compare'
        else:
            x_smooth = x
        if y_smooth in data.columns:
            if data_set=='':
                data_set = 'compare'
        else:
            y_smooth = y
            
        if data_set=='compare':
            raw, = plt.plot(
                data[x],
                data[y],
                'r-',
                linewidth = standard_linewidth,
                )
            smoothed, = plt.plot(
                data[x_smooth],
                data[y_smooth],
                'b-',
                linewidth = standard_linewidth,
                )
            plots = [smoothed, raw]
            labels = ['Smoothed data', 'Raw data']
            
        elif data_set=='raw_only' or data_set=='':
            plots, = plt.plot(
                data[x],
                data[y],
                'b-',
                linewidth = standard_linewidth,
                )
            labels = 'Raw data'
            
        elif data_set=='smooth_only':
            plots, = plt.plot(
                data[x_smooth],
                data[y_smooth],
                'b-',
                linewidth = standard_linewidth,
                )
            labels = 'Smoothed data'
            
        else:
            print('The given data_set argument is invalid.')
            
        return plots, labels


    # Create legend. Standard settings for plt.legend()

    def make_legend(plots, plot_labels):
        plt.legend(
            plots,
            plot_labels,
            facecolor = 'white',
            framealpha = 1
            )


    # Standard layout, and if-statements to show and/or export the figure as
    # necessary.

    def finalize_figure(figure_name):
        plt.tight_layout()
        if export_plots:
            file_formats = ['png', 'pdf'] # pdf needed for vector graphics.
            for ext in file_formats:
                file_name = figure_name + '.' + ext
                file_name = path.join(plot_directory, file_name)
                plt.savefig(
                    file_name,
                    format = ext,
                    dpi = 600
                    )
        if show_plots:
            plt.draw()
            plt.show(block=False)


    # This is a single function providing a simple interface for standard
    # graphs, as well as serving as an example of how the auxillary functions
    # above might be utilized.

    def plot_graph(figure_name, x, y, x_label, y_label, data_bank=''):
        figure_name = create_figure(figure_name)
        data_plots, data_labels = plot_data(
            x,
            y,
            data = data_bank
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        make_legend(data_plots, data_labels)
        finalize_figure(figure_name)

    
    # ========================= Specific plots ========================== #

    
    # ======================== Analogue sensors ========================= #
    
    # Pressure
    
    # figure_name = create_figure('Pressure')
    # plots, labels = plot_data(
    #     't',
    #     'pressure',
    #     )
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Pressure [kPa]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # Magnetic field (y)
    
    # figure_name = create_figure('Magnetic field (y)')
    # plots, labels = plot_data(
    #     't',
    #     'mag',
    #     )
    # plt.ylim(0, U_main)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('$M_y$ [V]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # Acceleration (y)
    
    # figure_name = create_figure('Acceleration (y)')
    # plots, labels = plot_data(
    #     't',
    #     'a_y',
    #     )
    # plt.ylim(-a_y_max, a_y_max)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('$a_y$ [gee]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # Acceleration (x)
    
    # figure_name = create_figure('Acceleration (x)')
    # plots, labels = plot_data(
    #     't',
    #     'a_x',
    #     )
    # plt.ylim(-a_x_max, a_x_max)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('$a_x$ [gee]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # Light sensor
    
    # figure_name = create_figure('Light sensor')
    # plot_data(
    #     't',
    #     'light',
    #     )
    # plt.ylim(0, U_main)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Brightness [V]')
    # finalize_figure(figure_name)


    # Internal temperature

    figure_name = create_figure('Internal temperature')
    plots, labels = plot_data(
        't',
        'temp_int',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    make_legend(plots, labels)
    finalize_figure(figure_name)


    # External temperature

    figure_name = create_figure('External temperature')
    plots, labels = plot_data(
        't',
        'temp_ext',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # A7
    
    if a7_occupied:
        figure_name = create_figure('Analogue channel 7')
        plot_data(
            't',
            'a7'
            )
        plt.xlabel('$t$ [s]')
        plt.ylabel('A7 [V]')
        finalize_figure(figure_name)



    # ======================== Temperature array ======================== #
    
    # Fancy plot
    
    # plt.rcParams['axes.grid'] = False # disables the grid. Remember to turn it back on again after this figure!
    # background_color = '#EAEAF2' # same color as the seaborn-theme. Not the most elegant solution
    # figure_name = 'Temperature array (smoothed)'
    # fig = plt.figure(
    #     figure_name,
    #     facecolor = background_color,
    #     clear = True,
    #     )
    # ax = plt.axes(facecolor = background_color)
    # sensor_IDs = [''.join(filter(str.isdigit, s)) for s in temp_array_channels]
    # im = ax.pcolormesh(
    #     temp_array['t'],
    #     sensor_IDs,
    #     temp_array[temp_array_channels_smooth].T,
    #     shading = 'nearest',
    #     cmap = 'hot',
    #     #vmin = 0, # Sets a lower limit on the temperature color bar
    #     #vmax = 40, # Sets an upper limit on the temperature color bar
    #     )
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Sensor ID')
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.ax.set_title(u'\N{DEGREE SIGN}C')
    # finalize_figure(figure_name)
    # plt.rcParams['axes.grid'] = True # enables the grid again for later plots


    # # Sanity check of fancy plot (used to verify data integrity)
    
    # figure_name = create_figure('Temperature array (sanity check)')
    # colormap = plt.cm.hsv
    # plt.gca().set_prop_cycle(plt.cycler('color', colormap(np.linspace(0, 1, len(temp_array_channels)))))
    # x = temp_array['t']
    # for i in temp_array_channels:
    #     y = temp_array[i]
    #     plt.plot(x, y, label=i, alpha=0.25)
    # for i in temp_array_channels_smooth:
    #     y = temp_array[i]
    #     plt.plot(x, y, label=i)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    # plt.legend(
    #     facecolor = 'white',
    #     framealpha = 1,
    #     ncol = 4,
    #     )
    # finalize_figure(figure_name)
    
    
    
    # ========================== Power sensor =========================== #
    
    # Battery voltage

    # figure_name = create_figure('Battery voltage')
    # if a7_occupied:
    #     plot_data(
    #         't',
    #         'voltage',
    #         data = power_sensor
    #         )
    # else:
    #     analogue_plot, = plt.plot(
    #         analogue['t'],
    #         analogue['a7'],
    #         '-',
    #         linewidth = standard_linewidth,
    #         )
    #     digital_plot, = plt.plot(
    #         power_sensor['t'],
    #         power_sensor['voltage'],
    #         '-',
    #         linewidth = standard_linewidth,
    #         )
    #     plots = [analogue_plot, digital_plot]
    #     labels = ['analogue', 'digital']
    #     make_legend(plots, labels)
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('[V]')
    # finalize_figure(figure_name)
    
    
    # # Current

    # figure_name = create_figure('Current')
    # plt.plot(
    #     power_sensor['t'],
    #     power_sensor['current'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Current [A]')
    # finalize_figure(figure_name)
    
    
    
    # =============================== GPS =============================== #
    
    # Speed (from GPS)

    # figure_name = create_figure('GPS speed')
    # plots, labels = plot_data(
    #     't',
    #     'speed',
    #     data = gps,
    #     )
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Speed [m/s]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # Height (from GPS)

    # figure_name = create_figure('GPS height')
    # plots, labels = plot_data(
    #     't',
    #     'height',
    #     data = gps,
    #     )
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Altitude [m]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    
    # =============================== IMU =============================== #
    
    # The IMU is a bit special, in that each channel is stored at different
    # time slots. Hence, we need to use unique time-variables for each channel.
    
    # IMU magnetometer
    
    # figure_name = create_figure('IMU mag')
    # x_plot, = plt.plot(
    #     imu['t_mag_x'],
    #     imu['mag_x'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # y_plot, = plt.plot(
    #     imu['t_mag_y'],
    #     imu['mag_y'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # z_plot, = plt.plot(
    #     imu['t_mag_z'],
    #     imu['mag_z'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # plots = [x_plot, y_plot, z_plot]
    # labels = ['$M_x$', '$M_y$', '$M_z$']
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Magnetic fieldstrength [gauss]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)
    
    
    # IMU gyroscope
    
    # figure_name = create_figure('IMU gyro')
    # x_plot, = plt.plot(
    #     imu['t_ang_vel_x'],
    #     imu['ang_vel_x'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # y_plot, = plt.plot(
    #     imu['t_ang_vel_y'],
    #     imu['ang_vel_y'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # z_plot, = plt.plot(
    #     imu['t_ang_vel_z'],
    #     imu['ang_vel_z'],
    #     '-',
    #     linewidth = standard_linewidth,
    #     )
    # plots = [x_plot, y_plot, z_plot]
    # labels = ['$x$', '$y$', '$z$']
    # plt.xlabel('$t$ [s]')
    # plt.ylabel('Angular velocity [degrees per second]')
    # make_legend(plots, labels)
    # finalize_figure(figure_name)


    # IMU accelerometer
    
    figure_name = create_figure('IMU accelerometer')
    x_plot, = plt.plot(
        imu['t_a_x_imu'],
        imu['a_x_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    y_plot, = plt.plot(
        imu['t_a_y_imu'],
        imu['a_y_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    z_plot, = plt.plot(
        imu['t_a_z_imu'],
        imu['a_z_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    plots = [x_plot, y_plot, z_plot]
    labels = ['$a_x$', '$a_y$', '$a_z$']
    plt.xlabel('$t$ [s]')
    plt.ylabel('Acceleration [gee]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    
    # ========================== Miscellaneous ========================== #
    
    # Frame counter

    figure_name = create_figure('Frame counter')
    plot_data(
        't',
        'framecounter',
        data = misc,
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Frame number')
    finalize_figure(figure_name)


# =========================================================================== #
# ========================== Export processed data ========================== #

# Much like before, a filedialog is opened, this time to allow the user to
# specify the name and storage location of the processed data.
# The processed data is stored using Pandas' .to_csv()

if export_processed_data or export_raw_data:
    print('Exporting ...')
    
    def export_frame(frame, frame_name, directory):
        if not path.exists(directory):
            os.mkdir(directory)
        data_file = parent_file_name + '_' + frame_name + '.csv'
        data_file = path.join(directory, data_file)
        frame.to_csv(data_file, sep = ',', decimal = '.', index = False)
    
    if export_processed_data:
        export_frame(analogue, 'analogue', processed_data_directory)
        export_frame(gps, 'GPS', processed_data_directory)
        export_frame(imu, 'IMU', processed_data_directory)
        export_frame(temp_array, 'temp_array', processed_data_directory)
        export_frame(power_sensor, 'power_sensor', processed_data_directory)
        export_frame(misc, 'misc', processed_data_directory)
    
    if export_raw_data:
        # Create an inverse channel dictionary
        raw_channels = {v: k for k, v in channels.items()}
        
        # Temporarily rename channel names back to their original names
        raw_data.rename(
            columns = raw_channels,
            inplace = True,
            )
        
        # Export raw data
        export_frame(raw_data, 'raw', working_directory)
        
        # Swap channel names back again
        raw_data.rename(
            columns = channels,
            inplace = True,
            )


# Create and export a kml-file which can be opened in Google Earth.

if export_kml:
    print('Exporting kml ...')

    # kml-coordinates needs to be in degrees for longitude and latitude, and
    # meters for the height. Hence, we will take our data from processed_data:
    if sanitize_gps:
        notna_indices = (
            gps['lat_smooth'].notna() &
            gps['long_smooth'].notna() &
            gps['height_smooth'].notna()
            )
        kml_lat = gps['lat_smooth'][notna_indices].copy().to_numpy()
        kml_long = gps['long_smooth'][notna_indices].copy().to_numpy()
        kml_height = gps['height_smooth'][notna_indices].copy().to_numpy()
    else:
        kml_lat = gps['lat'].copy().to_numpy()
        kml_long = gps['long'].copy().to_numpy()
        kml_height = gps['height'].copy().to_numpy()

    # To avoid having to install a kml-library, we will instead (ab)use a numpy
    # array and savetxt()-function to save our kml file.
    # Unfortunately, this means that we need to hard-code the kml-file ...
    kml_header = (
'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Document>
<name>Paths</name>
<description>Paths based on GPS/GNSS coordinates.</description>
<Style id="yellowLineGreenPoly">
<LineStyle>
<color>7f00ffff</color>
<width>4</width>
</LineStyle>
<PolyStyle>
<color>7f00ff00</color>
</PolyStyle>
</Style>
<Placemark>
<name>Student rocket path</name>
<description>Student rocket path, according to its onboard GPS</description>
<styleUrl>#yellowLineGreenPoly</styleUrl>
<LineString>
<extrude>0</extrude>
<tessellate>0</tessellate>
<altitudeMode>absolute</altitudeMode>
<coordinates>''')

    kml_body = np.array([kml_long, kml_lat, kml_height]).transpose()

    kml_footer = (
'''</coordinates>
</LineString>
</Placemark>
</Document>
</kml>''')

    data_file = parent_file_name + '.kml'
    data_file = path.join(working_directory, data_file)

    np.savetxt(
        data_file,
        kml_body,
        fmt = '%.6f',
        delimiter = ',',
        header = kml_header,
        footer = kml_footer,
        comments = '',
        )
