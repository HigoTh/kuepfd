import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import jn_zeros, j1, j0
from typing import List
from tqdm import tqdm
# Global parameters -----------------------------------------------------------

earth_radius = 6378.0 # Earth radius (km)
earth_mass = 5.9742*(10**24) # Earth mass (kg)
grav_const = 6.6743*(10**(-11)) # Gravitational constant
mu_c = earth_mass * grav_const
ref_bandwidth = 40e3    # Reference bandwidth (40 KHz)
glb_j1_zeros = jn_zeros( 1, 20 )

# Antenna patterns ------------------------------------------------------------

class EStAnt():

    def __init__(self, antenna_size: float, wavelength: float ):
        """ Definition of the earth station radiation pattern according to
        the ITU 1428.

        Args:
            antenna_size (float): Antenna size (m).
            wavelength (float): Wavelength (m).
        """
        self.antenna_size = antenna_size
        self.wavelength = wavelength

    def compute_ant_patt( self, phi_dg: float ):
        """ Calculation of Antenna Radiation Pattern According to ITU 1428.

        Args:
            phi_dg (float): Angle (degres).

        Returns:
            Antenna Gain (dBi).
        """
        d_lmb_r = self.antenna_size / self.wavelength
        # Apply abs() to account for radiation pattern symmetry
        phi_dg_p = np.abs( phi_dg )

        # Parameters
        g_max = 20.0 * np.log10( d_lmb_r ) + 7.7
        g_1 = 29.0 - 25.0 * np.log10( 95.0 / d_lmb_r )
        phi_m = ( 20.0 / d_lmb_r ) * np.sqrt( g_max - g_1 )

        # Angle ranges
        if 0.0 <= phi_dg_p < phi_m:
            return ( g_max - 2.5e-3 * ( d_lmb_r * phi_dg_p )**2 )
        
        if phi_m <= phi_dg_p < (95.0 / d_lmb_r):
            return g_1

        if (95.0 / d_lmb_r) <= phi_dg_p < 33.1:
            return 29.0 - 25.0 * np.log10( phi_dg_p )

        if (33.1) < phi_dg_p <= 80:
            return -9

        if (80.0) < phi_dg_p <= 180:
            return -5

        return -5

class NGSOAnt():

    glb_j1_zeros = jn_zeros( 1, 20 )
    num_pp_samples = 100000
    delta_th = (2 * np.pi) / num_pp_samples

    def __init__(self, 
                 antenna_size: float,
                 wavelength: float,
                 num_sec_lobes: int,
                 g_max: float,
                 slr: float ):
        """ Definition of the NGSO Antenna Radiation Pattern according to
        Design of Circular Apertures for Narrow Beamwidth and Low Sidelobes.

        Args:
            antenna_size (float): Antenna size (m).
            wavelength (float): Wavelength (m).
            num_sec_lobes (int): Number of secondary lobes.
            g_max (float): Maximum gain (dBi).
            slr (float): Secondary lobes ratio (dB).
        """
        self.antenna_size = antenna_size
        self.wavelength = wavelength
        self.num_sec_lobes = num_sec_lobes
        self.g_max = g_max
        self.slr = slr
        self.pp_ant_pattern = [ self.compute_ant_patt( thr ) for thr in np.linspace( -np.pi, np.pi, self.num_pp_samples ) ]

    def compute_ant_patt( self, th_rad ):

        # Bessel zeros
        mu_l = self.glb_j1_zeros[ 0 : self.num_sec_lobes - 1 ] / np.pi
        mu_lp = self.glb_j1_zeros[ self.num_sec_lobes - 1 ] / np.pi

        # Parameters
        ap = ( 1.0 / np.pi ) * np.arccosh( 10**( self.slr / 20.0 ) )
        sigma = mu_lp / np.sqrt( ap**2 + ( self.num_sec_lobes - 0.5 )**2 )

        # Observation position
        eps = 1e-10
        u = ( 2.0 / self.wavelength ) * self.antenna_size * np.sin( th_rad + eps )

        # Compute antenna radiation pattern ---------------------------------------
        i = np.arange( 1, self.num_sec_lobes )
        u_m = np.sqrt( ap**2 + ( i - 0.5 )**2 )
        k = ( 2 * j1( np.pi * u ) / ( np.pi * u ) ) * np.prod( ( 1.0 - ( u / ( sigma * u_m ) )**2 ) / ( 1.0 - ( u / ( mu_l ) )**2 ) )

        return self.g_max + 20.0 * np.log10( np.abs( k ) )

    def compute_ant_patt_pp( self, th_rad ):

        n = int( (th_rad + np.pi) / self.delta_th )
        return self.pp_ant_pattern[ n ]

def eval_interference_sp( num_sat_pp: int,
                       num_planes: int,
                       ngso_antenna: NGSOAnt,
                       ea_antenna: EStAnt,
                       f_pl_heigth: float,
                       pl_heigth_dh: float,
                       ngso_power_dbw: float,
                       ang_range: float,
                       epfd_th: float,
                       mit_angle: float,
                       steering_angle: float,
                       power_control_f: float,
                       num_s_samples: int = 1000 ):
    """

    Args:
        num_sat_pp (int): Number of satellites per plane.
        num_planes (int): Number of planes.
        ngso_antenna (NGSOAnt): NGSO antenna.
        ea_antenna (EStAnt): Earth station antenna.
        f_pl_heigth (float): First plane altitude.
        pl_heigth_dh (float): Plane difference altitude.
        ngso_power_dbw (float): NGSO power.
        ang_range (float): GSO coverage range.
        epfd_th (float): EPFD threshold.
        mit_angle (float): Mitigation angle zone.
        steering_angle (float): Steering angle.
        power_control_f (float): Power control coefficient.
        num_s_samples (int, optional): Number of spatial samples. Defaults to 1000.

    Returns:
        _type_: _description_
    """

    # NGSO power in watts
    ngso_power = 10**( ngso_power_dbw / 10.0 )

    # Initial angular positions of the satellites
    th_i = np.linspace(0, 2 * np.pi, num_sat_pp )

    # Earth station max antenna gain
    ea_st_max_gain = 10**( ea_antenna.compute_ant_patt( 0.0 ) / 10.0 )

    # Earth center as (0,0)
    earth_center = np.array([0, 0])
    ec_gso_vec = np.array([0, 1])

    # Allocation of the NGSO Satellites position matrix
    ngso_sat = np.zeros(( num_planes, num_sat_pp, 2 ))

    # Satellite height at first plane
    sat_h0 = f_pl_heigth + earth_radius

    # Earth station Position theta-axis
    eat_axis = np.linspace( np.pi/2 - ang_range, ang_range + np.pi / 2, num_s_samples )

    # EPFDs
    epfd_v = np.zeros( num_s_samples )
    epfd_mit = np.zeros( num_s_samples )

    # Number of planes
    for m in range(0, num_planes):
    
        # Phase between the plans
        phi_i = (np.pi / (2 * num_sat_pp)) * m

        # Satelitte height
        sat_h = ( sat_h0 + m * pl_heigth_dh )
        # Positioning all satellites in the plan
        for n in range( 0, num_sat_pp ):
            
            # Calculanting the position and angle of the satellites
            ngso_sat[m, n, 0] = np.cos(th_i[n] + phi_i + np.pi / 2) * sat_h
            ngso_sat[m, n, 1] = np.sin(th_i[n] + phi_i + np.pi / 2) * sat_h
            ngso_sat_angle = np.arccos(np.clip(np.dot(ec_gso_vec, ngso_sat[m, n, :]/ sat_h), -1.0, 1.0))

            for i, ea_th in enumerate( eat_axis ):

                earth_station = np.array([np.cos( ea_th ), np.sin( ea_th )]) * earth_radius
                # Vector between the earth station and the GSO satellite
                ea_gso_v = np.array([0, 42164]) - earth_station
                ea_gso_v = ea_gso_v / np.linalg.norm( ea_gso_v )

                # Vector between the earth station and all satellites
                ea_sat_v = earth_station - np.array(ngso_sat[m, n, : ])
                ea_sat_dist = np.linalg.norm( ea_sat_v ) # km
                # Normalizing the vetor between the earth station and satellite
                ea_sat_v = ea_sat_v / ea_sat_dist

                # Visibility condition
                if np.dot(ea_gso_v, -ea_sat_v) > 0.0:

                    # Compute angles ------------------------------------------
                    ec_sat_v = earth_center - np.array(ngso_sat[m, n, : ])
                    ec_sat_v = ec_sat_v / np.linalg.norm( ec_sat_v )

                    # Angle between the vector pointing to the Earth's center and one pointing to the earth station
                    ngso_th = np.arccos(np.clip(np.dot(ea_sat_v, ec_sat_v), -1.0, 1.0)) #rad
                    # Angle between the main beam of the ground station with respect to the NGSO satellite
                    ea_th = np.arccos(np.clip(np.dot(ea_gso_v, -ea_sat_v), -1.0, 1.0)) #rad

                    # Compute antenna gains -----------------------------------
                    g_rx_dbi = ea_antenna.compute_ant_patt( np.rad2deg( ea_th ) ) # Angle must be in degrees (°)
                    g_tx_dbi = ngso_antenna.compute_ant_patt_pp( ngso_th ) # Angle must be in rad
                    g_rx = 10**( g_rx_dbi / 10.0 )
                    g_tx = 10**( g_tx_dbi / 10.0 )

                    #Calculating the epfds of all visible satellites in the plan
                    sigle_sat_epfd = ( ( ngso_power / ref_bandwidth ) * ( g_tx / ( 4 * np.pi * ( ( ea_sat_dist * 1e3 )**2 ) ) ) * ( g_rx / ea_st_max_gain ) )
                    epfd_v[ i ] = epfd_v[ i ] + sigle_sat_epfd

                    if (ngso_sat_angle <= mit_angle/2.0):
                        
                        # Steering on the ngso antenna
                        g_tx_steering_dbi =  ngso_antenna.compute_ant_patt( ngso_th + steering_angle ) # Angle must be in rad
                        g_tx_steering = 10**( g_tx_steering_dbi / 10.0 )
                        # Joint steering and power control
                        epfd_mit[ i ] = epfd_mit[ i ] + sigle_sat_epfd * ( g_tx_steering / g_tx ) * power_control_f

                    else:
                        epfd_mit[ i ] = epfd_mit[ i ] + sigle_sat_epfd

    # Convert to dB
    epfd_v = 10.0 * np.log10( epfd_v )
    epfd_mit = 10.0 * np.log10( epfd_mit )

    # Compute the protected percentual
    prot_arc_w = ( np.sum( epfd_v < epfd_th ) / len( epfd_v ) )
    prot_arc_mit = ( np.sum( epfd_mit < epfd_th ) / len( epfd_mit ) )

    return eat_axis, epfd_v, epfd_mit, prot_arc_w, prot_arc_mit


if __name__ == "__main__":

    # Light speed
    c  = 2.998*(10**8)
    # Frequency
    f = 10.7 * (10**9)
    # Lambda
    lambd = c/f

    # Earth station antenna pattern -------------------------------------------
    
    # Antenna size
    ea_antenna_size = 0.2
    # Earth station antenna instance
    ea_antenna = EStAnt( ea_antenna_size, lambd )

    # Angle axis
    th = np.linspace( -np.pi / 2, np.pi / 2, 1000 )
    # Antenna pattern
    g_itu1428 = [ ea_antenna.compute_ant_patt( np.rad2deg( thp ) ) for thp in th ]

    # Plot
    f1 = plt.figure(1)
    plt.plot( np.rad2deg( th ), g_itu1428, label='ITU 1428' )
    plt.xlabel( 'Angle (°)' )
    plt.ylabel( 'Antenna pattern (dBi)' )

    # NGSO Satellite antenna pattern ------------------------------------------

    # NGSO antenna size
    ngso_antenna_size = 0.3
    # Number of secondary lobes
    ngso_num_sec_lobes = 5
    # Max. gain
    ngso_max_gain = 34.0
    # Side lobe ratio
    ngso_slr = 20.0
    # Antenna instance
    ngso_antenna = NGSOAnt( ngso_antenna_size, lambd, ngso_num_sec_lobes, ngso_max_gain, ngso_slr )

    # Antenna pattern
    g_itu1528 = [ ngso_antenna.compute_ant_patt_pp( thp ) for thp in th ]
    
    # Plot
    plt.plot( np.rad2deg( th ), g_itu1528, label='ITU 1528 (Modified)' )
    plt.xlabel( 'Angle (°)' )
    plt.ylabel( 'Antenna pattern (dBi)' )
    plt.title( 'Antenna radiation patterns' )
    plt.legend()
