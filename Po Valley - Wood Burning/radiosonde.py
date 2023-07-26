from datetime import datetime
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np
import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
import xarray as xr


dt = datetime(2022, 12, 26, 12)
# Start date of the simulation
start = datetime(2022, 11, 24, 00)
# Get interval between two timestamps as timedelta object
diff = dt - start
# Get interval between two timestamps in hours
a = int(diff.total_seconds() / 3600)
# San Pietro Capofiume Station
station = '16144' #16045


# Load data from chimere
ds = xr.open_dataset('D:/Tesi/OLD/PM/DatiPM/New/analisiPM25.nc')
pressure = xr.open_dataset('D:/Tesi/OLD/Analisi sinottiche/analisi.nc')
# Load data from wrf
wrfT = xr.open_dataset('D:/Tesi/OLD/Radiosonde/Awrf_t.nc')
wrfP = xr.open_dataset('D:/Tesi/OLD/Radiosonde/Awrf_P.nc')
wrfPB = xr.open_dataset('D:/Tesi/OLD/Radiosonde/Awrf_PB.nc')
# Load pbl data
wrf_pbl = xr.open_dataset('D:/Tesi/OLD/Radiosonde/Awrf_pblh.nc')

# Data extraction for given date (dt) and location (San Pietro Capofiume)
t_wrf = wrfT.T.sel(south_north=53).sel(west_east=103).sel(Time=a) + 300
p_wrf = wrfP.P.sel(south_north=53).sel(west_east=103).sel(Time=a)/100
pb_wrf = wrfPB.PB.sel(south_north=53).sel(west_east=103).sel(Time=a)/100
t_wrf = np.array(t_wrf)
p_wrf = np.array(p_wrf)
pb_wrf = np.array(pb_wrf)
tot_p = pb_wrf + p_wrf
t_wrf0 = []
for i in range(len(t_wrf)):
    t_wrf0.append(t_wrf[i]*(tot_p[i]/1000)**0.2854 - 273.15)
pblh_w = wrf_pbl.PBLH.sel(south_north=53).sel(west_east=103).sel(Time=a)
t_wrf0 = np.array(t_wrf0)
tot_p = np.array(tot_p)

# Constant for conversion
M = 0.02896968
T0 = 288.16
R0 = 8.314462618

# Function to convert altitude to pressure
def m_to_pa(meter):
    pa = 101325*np.exp(-9.81*M*meter/(T0*R0))
    return pa

def pa_to_m(pa):
    #m = -T0*R0/(9.81*M)*np.log(pa/102325) #- 82.795
    m = 4430*(1-(pa/pa[0])**(1/5.255))
    return m

pblh_w = m_to_pa(pblh_w)

# Read remote sounding data based on time (dt) and station
df = WyomingUpperAir.request_data(dt, station)

# Create dictionary of united arrays
data = pandas_dataframe_to_unit_arrays(df)

# Isolate united arrays from dictionary to individual variables
p = np.array(data['pressure'])
T  = np.array(data['temperature'])
Td = np.array(data['dewpoint'])
rh = np.array(data['relh'])
u = data['u_wind']
v = data['v_wind']
wind = data['speed']
h = data['height']

def to_potential_temp(temperature, pressure, reference_pressure=1000):
    # Calculation of potential temperature
    potential_temp = temperature #+ 273.15  # Convert temperature from Celsius to Kelvin
    potential_temp = potential_temp * (reference_pressure / pressure)**0.2854

    return potential_temp


def identify_inversion_layers(temp, height):
    inversion_layers = []
    temp_diff = np.diff(temp) / np.diff(height)
    in_inversion = False
    for i in range(len(temp_diff)):
        if temp_diff[i] > 0.005 / units.meter and not in_inversion:
            in_inversion = True
            base_height = height[i]
            base_temp = temp[i]
        elif temp_diff[i] <= 0.005/units.meter and in_inversion:
            in_inversion = False
            top_height = height[i]
            top_temp = temp[i]
            potential_temp_diff = top_temp - base_temp
            max_temp_diff = np.max(temp[i - int(len(temp_diff) * 0.05):i + 1] - base_temp)
            inversion_layers.append({
                'base_height': base_height,
                'top_height': top_height,
                'potential_temp_diff': potential_temp_diff,
                'max_temp_diff': max_temp_diff
            })
    return inversion_layers

def h80(temperature, height):
    temperature = pd.Series(temperature)
    temperature = temperature.rolling(3).mean()
    c = identify_inversion_layers(temperature,height)
    new = []
    for i in range(len(c)):
        if float(c[i]['potential_temp_diff']) > 2 :
            new.append(c[i])
            
    pblh = new[0]['top_height'] /units.meter
    
    if pblh > 4000:
        pblh = -9999
        maxi = 0
        for k in range(len(c)):
            if c[k]['max_temp_diff'] > maxi:
                if c[k]['top_height'] /units.meter < 4000:
                    pblh = c[k]['top_height'] /units.meter
                    maxi = c[k]['max_temp_diff']
    return pblh    

def identify_regime(theta_5, theta_2, delta_s=1):
    regime = None
    if (theta_5 - theta_2) < -delta_s:
        regime = "CBL"
    elif (theta_5 - theta_2) > delta_s:
        regime = "SBL"
    else:
        regime = "NRL"
    return regime

regime = identify_regime(to_potential_temp(T[4], p[4]), to_potential_temp(T[1], p[1]), 1)

def find_PBL_height(theta, height, delta_u=0.5, theta_dot_r=4, min_height=150):
    pbl_height = -9999
    k = None
    for i in range(len(height)):
        if height[i]/units.meter < min_height:
            continue
        if (theta[i] - theta[0] >= delta_u):
            k = i
            break
        if k is None:
            return pbl_height
        for i in range(k, len(height)):
            if (theta[i] - theta[i-1])/ (height[i] - height[i-1]) >= theta_dot_r:
                pbl_height = height[i]
                break
            return pbl_height
        
def find_PBL_height_SBL(potential_temperature_gradient, wind_speed, height):
    # Determine PBL height based on stability criteria
    pbl_height_stability = None
    for k in range(1, len(potential_temperature_gradient) - 2):
        if (potential_temperature_gradient[k] - potential_temperature_gradient[k-1] < -40 /units.kilometer and
            potential_temperature_gradient[k+1]*units.meter < 4 and
            potential_temperature_gradient[k+2]*units.meter < 4):
            pbl_height_stability = height[k]
            break

    # Determine PBL height based on wind shear criteria
    pbl_height_wind_shear = None
    for j in range(1, len(wind_speed) - 1):
        if (wind_speed[j] > wind_speed[j-1] and
            wind_speed[j] > wind_speed[j+1] and
            wind_speed[j] - wind_speed[j-1] >= 2 *units.meter/units.second and
            wind_speed[j] - wind_speed[j+1] >= 2 *units.meter/units.second and
            all(wind_speed[i] <= wind_speed[i+1] for i in range(j+1, len(wind_speed) - 1))):
            pbl_height_wind_shear = height[j]
            break

    # Return the lower of the two heights if both are found
    if pbl_height_stability is not None and pbl_height_wind_shear is not None:
        return min(pbl_height_stability, pbl_height_wind_shear)
    elif pbl_height_stability is not None:
        return pbl_height_stability
    elif pbl_height_wind_shear is not None:
        return pbl_height_wind_shear
    else:
        return -9999

def ll10(theta, height, wind_speed):
    potential_temperature_gradient = np.diff(theta) / np.diff(height)
    regime = identify_regime(theta[4], theta[1], 1)
    if regime == "SBL":
        pbl = find_PBL_height_SBL(potential_temperature_gradient, wind_speed, height)
    else:
        pbl = find_PBL_height(theta, height)
            
    return pbl

pbl_ll10 = ll10(to_potential_temp(T,p),h,wind)

###################################
#TKE

def virt_t(T,P,RH):
    T = T + 273.15
    L = 2260
    Rv = 461
    es = 611*np.exp(-L/Rv*(1/T-1/273.15))
    e = RH/100*es
    V = T * (1-(e/p)*(1-0.622))
    return V

def pbl_tke(T, p, v, u, rh):
    pbl = 0
    z = pa_to_m(p*100)
    V = virt_t(T, p, rh)
    theta_v0 = to_potential_temp(V[0], p[0])
    theta_vz = []
    for j in range(len(z)):
        theta_vz.append(to_potential_temp(V[j], p[j]))
        
    Rib = (9.81*z/theta_v0) * ((theta_vz - theta_v0)/((0.5144*u)**2 + (0.5144*v)**2))*units.knot**2
    
    Ric = 0.25
    i = np.nanargmin(abs(Rib-Ric))
    pbl = z[i]
    
    return pbl

# Change default to be better for skew-T
fig = plt.figure(figsize=(9, 11))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)
ra_t = [-450, 45]
pblh_w = [pblh_w/100, pblh_w/100]
pblh_tke = [m_to_pa(pbl_tke(T,p,v,u,rh))/100, m_to_pa(pbl_tke(T,p,v,u,rh))/100]

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
#skew.plot(p_mod, t_mod, 'orange', label="CHIMERE Temperature")
skew.plot(tot_p, t_wrf0, 'blue', label="WRF Temperature")
skew.plot(p, T, 'r', label="Temperature")
skew.plot(p, Td, 'g', label="Dew point")
skew.plot(pblh_w, ra_t, 'purple', label="WRF PBL Height")
skew.plot(pblh_tke, ra_t, 'orange', label="Rib 0.25")
#skew.plot([m_to_pa(pbl_ll10)/100, m_to_pa(pbl_ll10)/100], ra_t, 'grey', label="LL10")
skew.plot_barbs(p[::3], u[::3], v[::3], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y and create legend
skew.ax.set_xlim(-30, 40)
skew.ax.set_ylim(1020, 200)
skew.ax.legend(loc="upper right")

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                         alpha=0.25, color='tab:green')


# Add some descriptive titles
plt.title('San Pietro Capofiume Sounding', loc='left')
plt.title('Date: {}'.format(dt), loc='right')
font_axis = {'family': 'monospace',
    'color':  'black',
    'weight': 'normal',
    'size': 10,
    }
plt.xlabel("T [$^o$C]",fontdict=font_axis)
plt.ylabel("p [hPa]",fontdict=font_axis)
plt.savefig('C:/Users/manue/Desktop/skewT_Old4.png', dpi=500)
plt.show()









