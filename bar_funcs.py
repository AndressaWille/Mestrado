## funções para análise de galáxias barradas

from numpy import *
import numpy as np
import h5py

def shift_com(m, x, y, z):
    
    cm_x_disk = sum(m*x)/sum(m)
    cm_y_disk = sum(m*y)/sum(m)
    cm_z_disk = sum(m*z)/sum(m)

    x_new_disk = x - cm_x_disk
    y_new_disk = y - cm_y_disk
    z_new_disk = z - cm_z_disk
    
    return x_new_disk, y_new_disk, z_new_disk

def shift_com_2(m_disk, x_disk, y_disk, z_disk, m_halo, x_halo, y_halo, z_halo):

    cm_x_disk = sum(m_disk*x_disk)/sum(m_disk)
    cm_y_disk = sum(m_disk*y_disk)/sum(m_disk)
    cm_z_disk = sum(m_disk*z_disk)/sum(m_disk)

    x_new_disk = x_disk - cm_x_disk
    y_new_disk = y_disk - cm_y_disk
    z_new_disk = z_disk - cm_z_disk

    cm_x_halo = sum(m_halo*x_halo)/sum(m_halo)
    cm_y_halo = sum(m_halo*y_halo)/sum(m_halo)
    cm_z_halo = sum(m_halo*z_halo)/sum(m_halo)

    x_new_halo = x_halo - cm_x_halo
    y_new_halo = y_halo - cm_y_halo
    z_new_halo = z_halo - cm_z_halo
    
    return x_new_disk, y_new_disk, z_new_disk, x_new_halo, y_new_halo, z_new_halo


def bar_strength(m, x, y, Rmax, Nbins, n_snapshots):
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins

    r = np.empty(Nbins)
    ab = np.empty(Nbins)
    a0 = np.empty(Nbins)
    i2 = np.empty(Nbins)

    for i in range(0, Nbins):

        R1 = i * dR
        R2 = R1 + dR
        r[i] = (0.5 * (R1+R2))

        over = 1.0 * dR

        cond = np.argwhere((R > R1-over) & (R < R2+over)).flatten()

        a = m[cond] * np.cos(2*theta[cond])
        a_quad = sum(a)

        a_ = m[cond] * np.cos(0*theta[cond])
        a0[i] = sum(a_)

        b = m[cond] * np.sin(2*theta[cond])
        b_quad = sum(b)

        ab[i] = a_quad**2+b_quad**2

        i2[i] = np.sqrt(ab[i])/a0[i]
            

    a2 = max(i2)

    return a2


def v_circ(m_disk, m_halo, x_new_disk, y_new_disk, z_new_disk, x_new_halo, y_new_halo, z_new_halo, Rmax, Nbins):
    
    G = 43007.1

    R_disk = np.sqrt(x_new_disk**2 + y_new_disk**2 + z_new_disk**2)
    R_halo = np.sqrt(x_new_halo**2 + y_new_halo**2 + z_new_halo**2)

    M_r_disk = np.empty(Nbins)
    M_r_halo = np.empty(Nbins)
    M_r = np.empty(Nbins)
    r = np.empty(Nbins)
    v_c = np.empty(Nbins)
    v_c_disk = np.empty(Nbins)
    v_c_halo = np.empty(Nbins)

    Rmin = 0.0
    Rmax = Rmax
    Nbins = Nbins
    dR = (Rmax - Rmin)/Nbins
    
    for i in range(0, Nbins):
        R1 = i * dR
        R2 = R1 + dR
        r[i] = R2

        cond1 = np.argwhere(R_disk<=R2).flatten()
        M_r_disk[i] = sum(m_disk[cond1])

        cond2 = np.argwhere(R_halo<=R2).flatten()
        M_r_halo[i] = sum(m_halo[cond2])

        v_c_disk[i] = (np.sqrt(G*M_r_disk[i]/r[i]))
        v_c_halo[i] = (np.sqrt(G*M_r_halo[i]/r[i]))

        M_r[i] = (M_r_disk[i] + M_r_halo[i])

        v_c[i] = (np.sqrt(G*M_r[i]/r[i]))
    
    return v_c_disk, v_c_halo, v_c, r

def S(m_up, x_up, y_up, m_down, x_down, y_down, Rmax, Nbins, n_snapshots):
    #S = |A2(z>0) - A2(z<0)|
    
    a2_up = bar_strength(m_up, x_up, y_up, Rmax, Nbins, n_snapshots)
    a2_down = bar_strength(m_down, x_down, y_down, Rmax, Nbins, n_snapshots)
    
    S = abs(a2_up - a2_down)
    
    return S

def time_buckling(S1, time1):
    maxS = max(S1) 
    pos_max = np.where(S1 == maxS)
    time_max = float(time1[pos_max])
    
    return time_max