# -*- coding: utf-8 -*-
'''
This script refines raw data from a weather balloon. The
different data streams are separated and converted to physical units, stored,
filtered, processed and presented graphically for preliminary analysis.

The code presented below is meant to serve as a starting point for customized
data analysis, and should be expanded upon and edited as needed.

Andøya Space Education

Created on Tue Jan 25 2022 at 09:39:00.
Last modified [dd.mm.yyyy]: 16.01.2024
@author: bjarne.ådnanes.bergtun
Partly based on an earlier MATLAB code. The tkinter code is based on a script
by odd-einar.cedervall.nervik
'''

import tkinter as tk # GUI
import tkinter.filedialog as fd # file dialogs
import os # OS-specific directory manipulation
from os import path # common file path manipulations
import dask.dataframe as dd # import of data
import matplotlib.pyplot as plt # plotting
import matplotlib.ticker as tic # plotting (for easy tick modification)
import numpy as np # maths
import pandas as pd # data handling


# Setup of imported libraries

pd.options.mode.chained_assignment = None


# =========================================================================== #
# ========================= User defined parameters ========================= #

# Logical switches.
# If using Spyder, you can avoid needing to load the data every time by going
# to "Run > Configuration per file" and activating the option "Run in console's
# namespace instead of an empty one".

load_data = True
sanitize_GPS = True # Uses both satellite number and approximate speed limits
convert_data = True
process_data = True
create_plots = False
show_plots = False
telemetry_plots = False # Plot housekeeping sensors and telemetry calculations?
export_plots = False # Note that plots cannot be exported unless created!
export_data = True
export_kml = False


# Approximate speed limits used in GPS sanitization.
# For the ground speed, it should be notated that north--south and east--west
# is checked independently, so the actually allowed maximal ground speed will
# be between 1 to sqrt(2) times larger than the specified value, depending on
# direction.

max_vertical_speed = 150 # [m/s]
max_ground_speed = 300 # [m/s]


# Quantization parameters

U_main = 3.3
standard_wordlength = 12
dt_standard = 4 # [s]


# telemetry link budget parameters

P_tx = 20 # Transmitter power, measured in dBm
G = 3 # Receiver antenna gain, measured in dBi
f = 433.5e6 # Carrier frequency, measured in Hz


# NTC parameters
# R_NTC should be the NTC reference resistance in ohms, e.g. R_NTC_ext = 470.

R_fixed_ext = 18e3
R_NTC_ext = 1e3

R_fixed_int = 6.8e3
R_NTC_int = 1e3


# Antenna position parameters (pro tip: these can be found using Google Maps
# or a GPS unit). The antenna is assumed to be stationary.

antenna_lat = 69.29597
antenna_long = 16.03046
antenna_height = 9 # height above ground in m.

# Channel spesification & naming.
# The channels will be named according to this list, so both numbering and
# ordering of items must be in accordance with the data file!

general = [
    't',
    'framecounter',
    ]

gps = [
    'lat',
    'long',
    'height',
    'GPS_satellites',
    ]

sensors = [
    'temp',
    'voltage',
    'temp_int',
    'pressure',
    'humidity_rel',
    ]

telemetry = [
    'RSSI',
    'CRC'
    ]

channels = general + gps + sensors + telemetry


# =========================================================================== #
# ============================== Load CSV data ============================== #

if load_data:

    # First a root window is created and put on top of all other windows.

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    # On top of the root window, a filedialog is opened to get the CSV file
    # from the file explorer.

    data_file = fd.askopenfilename(
        title = 'Select weather balloon data to import',
        filetypes = (
            ('ASE balloon files','.dat .log'),
            ('All files','.*')
            ),
        parent = root
        )
    
    print('\nFile path:\n',data_file,'\n')

    # Save some file paths for later.

    parent_file_name, parent_file_extension = path.splitext(
        path.basename(data_file)
        )
    working_directory = ''

    
    # Number of rows to skip depends on file type.
    if parent_file_extension == '.dat': # Note that dat-files from Education use utf-8!
        skip_rows = 0
        skip_footer = 0

    elif parent_file_extension == '.log':
        skip_rows = 3
        skip_footer = 3
        
    else:
        print('Unexpected file type. Assuming no skippable rows.')
        skip_rows = 0
        skip_footer = 0


    # Use dask to load the file, saving only the lines with t >= t_0 into
    # memory, and only the colons listed in the channels dictionary defined
    # above.

    raw_data = dd.read_csv(
        data_file,
        sep = ',',
        header = None,
        skiprows = skip_rows,
        skipfooter = skip_footer,
        names = channels,
        dtype = 'float64',
        na_values = ' nan',
        engine = 'python',
        )
    raw_data = raw_data[raw_data['CRC']>=1]
    raw_data[sensors] = raw_data[sensors][raw_data['CRC']>=2]
    raw_data[gps] = raw_data[gps][raw_data['CRC']!=2]
    raw_data = raw_data.compute()

    # Sanitize GPS-data
    if sanitize_GPS:
        mask = raw_data['GPS_satellites'] < 3
        mask2 = raw_data['GPS_satellites'] == 3
        mask3a = raw_data['lat'] <= 0
        mask3b= raw_data['lat'] > 90
        mask4a = raw_data['long'] <= 0
        mask4b = raw_data['long'] > 360
        raw_data.loc[mask, ['lat', 'long', 'height']] = np.nan
        raw_data.loc[mask2, 'height'] = np.nan
        raw_data.loc[mask3a, ['lat', 'long', 'height']] = np.nan
        raw_data.loc[mask3b, ['lat', 'long', 'height']] = np.nan
        raw_data.loc[mask4a, ['lat', 'long', 'height']] = np.nan
        raw_data.loc[mask4b, ['lat', 'long', 'height']] = np.nan

    # For the sake of plotting, we want to insert one nan-value every time we
    # have missing data. Since the PTU is sending at a regular interval
    # (specified above as dt_standard), missing data can be identified by
    # looking at the change in time, t.diff().
    dt = raw_data['t'].diff()

    dt_max = dt_standard*1.5 # Let's introduce a bit of tolerance

    # Copy the lines where missing data needs to be inserted.
    missing_data = raw_data.loc[dt > dt_max].copy()

    # Create column of missing data, and fill the missing data to be inserted
    nan_column = np.full(len(missing_data['t']), np.nan)
    for i in channels:
        if i != 't':
            missing_data[i] = nan_column

    # When dt is larger than dt_standard, this means that a missing data line
    # needs to be inserted *before* the current line.
    missing_data['t'] -= dt_standard
    missing_data.index -= 0.5

    # Insert the missing data, and sort according to the index.
    raw_data = pd.concat([raw_data, missing_data]).sort_index()

    # The indices needs to be reset for iloc[] to work as expected.
    raw_data.reset_index(drop=True, inplace=True)


# =========================================================================== #
# ============================== Convert data =============================== #

if convert_data:

    print('Converting data to sensible units ...')

    # Physical constants

    T_0 = 273.15 # 0 celsius degrees in kelvin


    # Steinhart--Hart parameter dictionary. This dictionary simplifies the
    # needed user input when using different NTC-sensors. Make sure it is up to
    # date!

    SH_dictionary = {
        10e3: (
            3.354016e-3,
            2.569850e-4,
            2.620131e-6,
            6.383091e-8
            ),
        1e3: (
            3.354016e-3,
            2.909670e-4,
            1.632136e-6,
            7.192200e-8
            ),
        '1k_above25': (
            3.354016e-3,
            2.933908e-4,
            3.494314e-6,
            -7.71269e-7
            ),
        470: (
            3.354075e-3,
            2.972812e-4,
            3.629995e-6,
            1.977197e-7
            )
        }


    # Conversion formulas

    def volt(bit_value, wordlength, U=U_main):
        Z = 2**wordlength - 1
        return U*bit_value/Z

    def NTC(U_bit, R_ref, R_fixed, wordlength=standard_wordlength): # unit: celsius degrees
        divisor = 2**wordlength-1-U_bit
        divisor[divisor<=0] = np.nan # avoids division by zero
        R_rel = R_fixed/R_ref
        R = R_rel*U_bit/divisor
        R[R<=0] = np.nan # avoids complex logarithms
        ln_R = np.log(R)
        A_1, B_1, C_1, D_1 = SH_dictionary[R_ref]
        T = 1/(A_1 + B_1*ln_R + C_1*ln_R**2 + D_1*ln_R**3) - T_0
        # The parameters for the 1k NTC resistor changes above 25 celsius:
        if R_ref == 1e3:
            A_1, B_1, C_1, D_1 = SH_dictionary['1k_above25']
            T_2 = 1/(A_1 + B_1*ln_R + C_1*ln_R**2 + D_1*ln_R**3) - T_0
            T[T>25] = T_2[T>25]
        return T

    def power(U_bit): # unit: volts
        U = volt(U_bit,8)
        return 2*U
    
    
    # Convert data units

    processed_data = raw_data.copy()

    processed_data['t'] = raw_data['t'] - raw_data['t'].iloc[0]

    processed_data['height'] = raw_data['height']/1000 # Converts to km
    processed_data['pressure'] = raw_data['pressure']/1000 # Converts to kPa

    processed_data['temp'] = NTC(
        raw_data['temp'],
        R_NTC_ext,
        R_fixed_ext
        )
    processed_data['temp_int'] = NTC(
        raw_data['temp_int'],
        R_NTC_int,
        R_fixed_int,
        wordlength=8
        )
    processed_data['voltage'] = power(raw_data['voltage'])


# =========================================================================== #
# ============================= Processes data ============================== #

if process_data:

    print('Calculating useful stuff ...')
    
    
    # ============================= Utility ============================= #

    # Smoothing function

    def smooth(data, r_tol=0.02, tol=np.nan):
        """
        Smooths a data series by requiring that the change between one datapoint and the next is below a certain relative treshold (the default is 2 % of the total range of the values in the dataseries). If this is not the case, the value is replaced by NaN. This gives a 'rougher' output than a more sophisticated smoothing algorithm (see for example statsmodels.nonparametric.filterers_lowess.lowess from the statmodels module), but has the advantage of being very quick. If more sophisticated methods are needed, this algorithm can be used to trow out obviously erroneous data before the data is sent through a more traditional smoothing algorithm.

        Parameters
        ----------
        data : pandas.DataSeries
            The data series to be smoothed

        r_tol : float, optional
            Tolerated change between one datapoint and the next, relative to the full range of values in DATA. The default is 0.02.
            
        tol : float, optional
            Absolute tolerated difference between interpolated datapoints. Ignored unless specified.

        Returns
        -------
        data_smooth : pandas.DataSeries
            Smoothed data series.

        """
        if tol==np.nan:
            valid_data = data[data.notna()]
            if len(valid_data) == 0:
                data_range = np.inf
            else:
                data_range = np.ptp(valid_data)
            tol = r_tol*data_range
        data_smooth = data.copy()
        data_interpol = data_smooth.interpolate()
        data_smooth[np.abs(data_interpol.diff()) > tol] = np.nan
        return data_smooth
    

    # ========== Coordinate transformations & antenna distance ========== #

    # Constants used in WGS 84

    r_a = 6378137. # Semi-major axis in meters.
    r_b = 6356752.314245 # Approximate semi-minor axis in meters.


    # Step 0: Convert angles to radians, as this makes for easier calculations

    processed_data['lat'] = np.radians(raw_data['lat'])
    processed_data['long'] = np.radians(raw_data['long'])
    
    antenna_lat = np.radians(antenna_lat)
    antenna_long = np.radians(antenna_long)


    # Step 1: Find spherical coordinates relative Earth's centre

    def r_E(latitude, height): # height must be in meters!
        r_a_cos = r_a * np.cos(latitude)
        r_b_sin = r_b * np.sin(latitude)
        return np.sqrt(r_a_cos**2 + r_b_sin**2) + height # distance to Earth's centre in meters


    # Step 2: Convert from spherical to cartesian coordinates, with axes
    # oriented such that conversion is easy -- z pointing north and x pointing
    # from the center of the Earth to the Greenwich meridian.

    def cartesian(latitude,longitude, height): # height must be in meters!
        r = r_E(latitude, height)
        r_cos = r * np.cos(latitude)
        x = r_cos * np.cos(longitude)
        y = r_cos * np.sin(longitude)
        z = r * np.sin(latitude)
        return (x, y, z)

    x_antenna, y_antenna, z_antenna = cartesian(
        antenna_lat, # radians
        antenna_long, # radians
        antenna_height # meters
        )


    # Step 3: Calculate distance using pytagoras

    def antenna_distance(latitude,longitude,height): # height must be in meters!
        x, y, z = cartesian(latitude,longitude, height)
        Dx = x - x_antenna
        Dy = y - y_antenna
        Dz = z - z_antenna
        r = np.sqrt(Dx**2 + Dy**2 + Dz**2)
        return r


    # Implementation: First sanitization, then calculation.
    
    if sanitize_GPS:
        lat_tolerance = max_ground_speed * dt_standard / r_E(antenna_lat, 0)
        long_tolerance = lat_tolerance / np.cos(antenna_lat)
        height_tolerance = max_vertical_speed * dt_standard / 1000
        processed_data['lat'] = smooth(
            processed_data['lat'],
            tol=lat_tolerance,
            )
        processed_data['long'] = smooth(
            processed_data['long'],
            tol=long_tolerance,
            )
        processed_data['height'] = smooth(
            processed_data['height'],
            tol=height_tolerance,
            )
    
    
    processed_data['r'] = antenna_distance(
        processed_data['lat'], # radians
        processed_data['long'], # radians
        raw_data['height'] # meters
        )/1000 # Unit: km


    # ===================== Speed, wind & direction ===================== #

    # When calculating wind speed, we are interested in the movement parallel
    # and perpendicular to the Earth's surface. Hence, we will be using
    # localized coordinates, with the z-axis always pointing upwards (towards
    # increasing height), and the x-axis pointing northwards (towards incre-
    # asing latitude). Hence, the y-axis will be pointing westwards (towards
    # decreasing longitude).
    # Obviously, this choice of directions will not work at the poles, but it
    # is highly unlikely that our weather balloon will end up at either pole.

    # Useful constants

    tau = np.pi*2


    # We start by calculating the horizontal speed, as this is straigthforward.

    def speed(position, time):
        dx = position.diff()
        dt = time.diff()
        return dx / dt

    processed_data['v_z'] = speed(
        raw_data['height'], # meters
        processed_data['t'] # seconds
        )


    # Then we turn to the wind.

    # To get sensible plots of wind direction,* we need to remove (or at least
    # minimize) the discontinuities caused by the multi-valued nature of
    # arctan(). The simplest way to do this, and what we will do, is to ensure
    # that the "jump" from one measurement to the next never exceeds pi.
    # -----------------
    # *This is only needed if we use cartesian plots. If we instead use
    #  spherical plots, this sanitizing is not necessary.

    def sanitize_wind_direction(direction_data):
        increments = direction_data.diff()
        for i in np.arange(len(direction_data)-1):
            if increments.iloc[i] > np.pi:
                direction_data.iloc[i] -= tau
                increments.iloc[i+1] += tau
            elif increments.iloc[i] < -np.pi:
                direction_data.iloc[i] += tau
                increments.iloc[i+1] -= tau
        i += 1
        if increments.iloc[i] > np.pi:
            direction_data.iloc[i] -= tau
        elif increments.iloc[i] < -np.pi:
            direction_data.iloc[i] += tau
        return direction_data

    # The wind direction is defined according to where the wind is coming
    # from, with 0 degrees being North, and 90 degrees being East.
    # Hence the direction is opposite of the direction of travel, and measured
    # clockwise rather than anti-clockwise.
    # The wind direction is measured in radians, while it's speed is in m/s.

    def horizontal_velocity(latitude,longitude,height,time): # height must be in meters!
        r = r_E(latitude,height)
        dt = time.diff()
        dphi = latitude.diff()
        dtheta = longitude.diff()
        dx = r * dphi
        dy = -r * dtheta
        speed = np.sqrt(dx**2 + dy**2)/dt
        direction = sanitize_wind_direction(-np.arctan2(-dy,-dx))
        return (speed, direction)

    processed_data['wind'], processed_data['wind_direction'] = horizontal_velocity(
        processed_data['lat'], # radians
        processed_data['long'], # radians
        raw_data['height'], # meters
        processed_data['t'] # seconds
        )
    
    processed_data['wind_direction_deg'] = np.rad2deg(processed_data['wind_direction'])


    # ================= Other atmospherical calculations ================ #

    # Constants used in the Arden Buck equation

    a = 6.1115 # Unit: hPa
    b = 23.036
    c = 279.82 # Unit: degrees celsius
    d = 333.7 # Unit: degrees celsius

    # Auxillary function used in the Arden Buck equation
    def temp_term(temp):
        return (b-temp/d)*(temp/(c+temp))

    # Vapor pressure
    def vapor_pressure(temp, hum_rel):
        return a*hum_rel*np.exp(temp_term(temp))/1000 # Unit: kPa

    # Dew point
    def dew_point(temp, hum_rel):
        humidity = hum_rel/100
        humidity[humidity<=0] = np.nan # avoids complex logarithms
        gamma = np.log(humidity) + temp_term(temp)
        return c*gamma/(b-gamma)


    processed_data['vapor_pressure'] = vapor_pressure(
        processed_data['temp'],
        processed_data['humidity_rel']
        )

    processed_data['dew_point'] = dew_point(
        processed_data['temp'],
        processed_data['humidity_rel']
        )


    # ==================== Link analysis (telemetry) ==================== #

    # Useful constants

    c = 299792.458 # km/s
    C_0 = 20 * np.log10(4 * np.pi * f/c)


    # Distance vector

    r_min = processed_data['r'].min() # km
    r_max = processed_data['r'].max() # km
    n = 10000 # number of data points
    r = np.linspace(r_min, r_max, n)


    # Free path loss

    L_0 = C_0 + 20 * np.log10(r)


    # Extremely simplified link budget

    P_rx = P_tx + G - L_0


    # ===================== Identify balloon burst ====================== #

    # Identify the point at which the balloon bursts, and split the data set
    # accordingly
    
    if raw_data['height'].max() > 0:
        burst_index = raw_data['height'].idxmax()
    else:
        burst_index = -1

    up_data = processed_data.iloc[:burst_index]
    down_data = processed_data.iloc[burst_index:]


# =========================================================================== #
# =========================== Prepare for export ============================ #

export = export_data or export_kml or export_plots

if export and working_directory == '':
    working_directory = fd.askdirectory(
        title = 'Choose output folder',
        parent = root
        )
    plot_directory = path.join(working_directory, 'Plots')


# =========================================================================== #
# ================================ Plot data ================================ #

if create_plots:

    print('Plotting ...')

    plt.ioff() # Prevent figures from showing unless calling plt.show()
    plt.style.use('seaborn-v0_8') # plotting style.
    plt.rcParams['legend.frameon'] = 'True' # Fill the background of legends.

    if export_plots and not path.exists(plot_directory):
        os.mkdir(plot_directory)


    # ==================== Custom plotting functions ==================== #

    # Custom parameters

    standard_linewidth = 0.5


    # Below is some auxillary functions containing some often-needed lines of
    # code for custom plots.


    # Create a figure, or ready an already existing figure for new data.
    # Returns the window title for easier export with finalize_figure().

    def create_figure(name):
        name = name + ' [' + parent_file_name + ']'
        plt.figure(name, clear=True)
        return name


    # Plot a single data set, with data from the descension plotted behind
    # those from the ascension.

    def plot_data(x, y, label=''):
        if label != '':
            down_label = label + ' (descending)'
            up_label = label + ' (ascending)'
        else:
            down_label = 'Descending'
            up_label = 'Ascending'
        down, = plt.plot(
            down_data[x],
            down_data[y],
            'r-',
            linewidth=standard_linewidth,
            )
        up, = plt.plot(
            up_data[x],
            up_data[y],
            'b-',
            linewidth=standard_linewidth,
            )
        plots = [up, down]
        labels = [up_label, down_label]
        return plots, labels


    # Similar to plot_data(), but this function allows plotting two data sets.
    # Data set 2 will be plotted behind data set 1, but the legend will order
    # the data sets according to whether the balloon is going up- or downwards.

    def plot_data_x2(data1_label, x1, y1, data2_label, x2, y2):
        down1_label = data1_label + ' (descending)'
        down2_label = data2_label + ' (descending)'
        up1_label = data1_label + ' (ascending)'
        up2_label = data2_label + ' (ascending)'
        down2, = plt.plot(
            down_data[x2],
            down_data[y2],
            'c-',
            linewidth=standard_linewidth,
            )
        up2, = plt.plot(
            up_data[x2],
            up_data[y2],
            'g-',
            linewidth=standard_linewidth,
            )
        down1, = plt.plot(
            down_data[x1],
            down_data[y1],
            'r-',
            linewidth=standard_linewidth,
            )
        up1, = plt.plot(
            up_data[x1],
            up_data[y1],
            'b-',
            linewidth=standard_linewidth,
            )
        plots = [up1, up2, down1, down2]
        labels = [up1_label, up2_label, down1_label, down2_label]
        return plots, labels

    
    # A simple function for labeling

    def label_figure(x_label, y_label='', plots='', plot_labels=''):
        plt.xlabel(x_label)
        if y_label != '':
            plt.ylabel(y_label)
        if plots != '':
            plt.legend(plots, plot_labels, facecolor='white', framealpha=1)


    # Plot labels and the legend (if arguments are given), and reveal and/or
    # export the figure

    def finalize_figure(figure_name, x_label='', y_label='', plots='', plot_labels=''):
        if x_label != '':
            label_figure(x_label, y_label, plots, plot_labels)
        plt.tight_layout()
        if export_plots:
            file_formats = ['png', 'pdf'] # pdf needed for vector graphics.
            for ext in file_formats:
                file_name = figure_name + '.' + ext
                file_name = path.join(plot_directory, file_name)
                plt.savefig(
                    file_name,
                    format = ext,
                    dpi = 300
                    )
        if show_plots:
            plt.draw()
            plt.show(block=False)


    # This is a single function providing a simple interface for standard
    # graphs, as well as serving as an example of how the auxillary functions
    # above might be utilized.

    def plot_graph(figure_name, x, y, x_label, y_label, data_label=''):
        figure_name = create_figure(figure_name)
        data_plots, data_labels = plot_data(
            x,
            y,
            label = data_label
            )
        finalize_figure(
            figure_name,
            x_label,
            y_label,
            data_plots,
            data_labels,
            )
        
    
    
    # An auxillary function for formatting azimuth tick labels.
    # Apply on the x-labels of an axis object by writing:
    #       ax.xaxis.set_major_formatter(tic.FuncFormatter(azimuth_ticks))
    # where ax is the axis object in question.
    
    def azimuth_ticks(x, pos):
        if x%360 == 0:
            return 'N'
        elif x%360 == 90:
            return 'E'
        elif x%360 == 180:
            return 'S'
        elif x%360 == 270:
            return 'W'
        else:
            return str(x) + u'\N{DEGREE SIGN}'
        


    # ========================= Specific plots ========================== #
    
    # Wind
    
    figure_name = create_figure('Wind')
    # Vertical speed
    ax1 = plt.subplot(131)
    line = ax1.axvline(
        x = 5,
        color = 'k',
        linestyle = '--',
        linewidth = 0.85
        )
    data_plot, data_label = plot_data(
        'v_z',
        'height',
        )
    plots = data_plot + [line]
    plot_labels = data_label + ['Nominal ascension speed']
    label_figure('Vertical speed [m/s]', 'Height [km]')
    # Horizontal speed
    ax2 = plt.subplot(132, sharey=ax1)
    plot_data(
        'wind',
        'height',
        )
    label_figure('Horizontal speed [m/s]')
    # Horizontal direction
    ax3 = plt.subplot(133, sharey=ax1)
    plot_data(
        'wind_direction_deg',
        'height',
        )
    ax3.xaxis.set_major_formatter(tic.FuncFormatter(azimuth_ticks))
    label_figure('Horizontal direction', '', plots, plot_labels)
    finalize_figure(figure_name)
    

    # # Vertical speed v. height

    # figure_name = create_figure('Vertical speed')
    # line = plt.axvline(
    #     x = 5,
    #     color = 'k',
    #     linestyle = '--',
    #     linewidth = 0.85
    #     )
    # data_plot, data_label = plot_data(
    #     'v_z',
    #     'height'
    #     )
    # plots = data_plot + [line]
    # plot_labels = data_label + ['Nominal ascension speed']
    # finalize_figure(
    #     figure_name,
    #     'Vertical speed [m/s]',
    #     'Height [km]',
    #     plots,
    #     plot_labels
    #     )


    # # Wind speed v. height

    # plot_graph(
    #     'Wind speed',
    #     'wind',
    #     'height',
    #     'Wind speed [m/s]',
    #     'Height [km]'
    #     )


    # Wind direction v. height
    # We will use a fancy polar plot for this.

    figure_name = create_figure('Wind direction (polar plot)')
    ax = plt.subplot(111, polar=True)
    data_plots, data_labels = plot_data(
        'wind_direction',
        'height',
        label = 'Wind direction',
        )
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        (0, 30, 60,
         90, 120, 150,
         180, 210, 240,
         270, 300, 330),
        labels = (
            'N',
            u'30\N{DEGREE SIGN}',
            u'60\N{DEGREE SIGN}',
            'E',
            u'120\N{DEGREE SIGN}',
            u'150\N{DEGREE SIGN}',
            'S',
            u'210\N{DEGREE SIGN}',
            u'240\N{DEGREE SIGN}',
            'W',
            u'300\N{DEGREE SIGN}',
            u'330\N{DEGREE SIGN}'
            )
        )
    legend_angle = np.radians(53)
    plt.legend(
        data_plots,
        data_labels,
        facecolor = 'white',
        framealpha = 1,
        loc = 'lower left',
        bbox_to_anchor = (
            0.555 + np.cos(legend_angle)/2,
            0.555 + np.sin(legend_angle)/2,
            )
        )
    label_position=ax.get_rlabel_position()
    ax.text(
        np.radians(label_position+10),
        .6*ax.get_rmax(),
        'Height [km]',
        rotation=80-label_position,
        ha='center',
        va='center',
        )
    finalize_figure(figure_name)


    # Pressure v. height

    plot_graph(
        'Pressure',
        'pressure',
        'height',
        'Pressure [kPa]',
        'Height [km]'
        )


    # Temperature v. height

    figure_name = create_figure('Temperature')
    data_plots, data_labels = plot_data_x2(
        'Temperature',
        'temp',
        'height',
        'Dew point',
        'dew_point',
        'height'
        )
    finalize_figure(
        figure_name,
        u'Temperature [\N{DEGREE SIGN}C]',
        'Height [km]',
        data_plots,
        data_labels
        )


    # Humidity v. height

    plot_graph(
        'Humidity',
        'humidity_rel',
        'height',
        'Relative humidity [%]',
        'Height [km]'
        )



    if telemetry_plots:

        # Internal temperature v. height
    
        plot_graph(
            'Internal temperature',
            'temp_int',
            'height',
            u'Internal temperature [\N{DEGREE SIGN}C]',
            'Height [km]'
            )
    
    
        # Battery
    
        plot_graph(
            'Battery',
            't',
            'voltage',
            '$t$ [s]',
            'Battery voltage [V]'
            )
    
    
        # Frame counter
    
        plot_graph(
            'Frame counter',
            't',
            'framecounter',
            '$t$ [s]',
            'Frame number'
            )
    
    
        # Distance v. time
    
        plot_graph(
            'PTU–antenna distance',
            't',
            'r',
            '$t$ [s]',
            'Distance between PTU and antenna [km]'
            )
    
    
        # RSSI:
    
        figure_name = create_figure('RSSI')
        theoretical_plot, = plt.plot(
            r,
            P_rx,
            'k--',
            linewidth = 0.85
            )
        data_plot, data_label = plot_data(
            'r',
            'RSSI'
            )
        plots = data_plot + [theoretical_plot]
        plot_labels = data_label + ['Theoretical curve']
        finalize_figure(
            figure_name,
            'Distance between PTU and antenna [km]',
            'Received power [dBm]',
            plots,
            plot_labels
            )
        
    


# =========================================================================== #
# ========================== Export processed data ========================== #

if export_data:
    print('Exporting processed data ...')

    data_file = parent_file_name + '.csv'
    data_file = path.join(working_directory, data_file)

    processed_data.to_csv(
        data_file,
        sep = ';',
        decimal = ',',
        index = False)


# Create and export a kml-file which can be opened in Google Earth.

if export_kml:
    print('Exporting kml ...')

    # kml-coordinates needs to be in degrees for longitude and latitude, and
    # meters for the height. Hence, we will take our data from raw_data:
    notna_indices = (
        raw_data['lat'].notna() &
        raw_data['long'].notna() &
        raw_data['height'].notna()
        )
    kml_lat = raw_data['lat'][notna_indices].copy().to_numpy()
    kml_long = raw_data['long'][notna_indices].copy().to_numpy()
    kml_height = raw_data['height'][notna_indices].copy().to_numpy()

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
<name>Balloon path</name>
<description>Weather balloon path, according to its onboard GPS.</description>
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
        comments = ''
        )

plt.show() if create_plots else None