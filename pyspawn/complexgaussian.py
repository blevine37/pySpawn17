# this module contains functions for doing complex Gaussian math.  Right
# now everything is hard coded for adiabatic/diabatic representation, but
# it shouldn't be hard to modify for DGAS

import sys
import types
import math
import cmath
import numpy as np
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj

def overlap_nuc_elec(ti, tj,positions_i="positions", positions_j="positions",\
                     momenta_i="momenta", momenta_j="momenta"):
    """Compute the overlap of two virbonic TBFs (electronic part included)"""
    
    if ti.istate == tj.istate:
        Sij = overlap_nuc(ti, tj, positions_i=positions_i, positions_j=positions_j,\
                          momenta_i=momenta_i, momenta_j=momenta_j)
    else:
        Sij = complex(0.0,0.0)
    return Sij

def overlap_nuc(ti, tj,positions_i="positions", positions_j="positions",\
                momenta_i="momenta", momenta_j="momenta"):
    """Compute the overlap of two nuclear TBFs (electronic part not included)"""
    
    if isinstance(positions_i, types.StringTypes):
        ri = eval("ti." + positions_i)
    else:
        ri = positions_i
    if isinstance(positions_j, types.StringTypes):
        rj = eval("tj." + positions_j)
    else:
        rj = positions_j
    if isinstance(momenta_i, types.StringTypes):
        pi = eval("ti." + momenta_i)
    else:
        pi = momenta_i
    if isinstance(momenta_j, types.StringTypes):
        pj = eval("tj." + momenta_j)
    else:
        pj = momenta_j
        
    widthsi = ti.widths
    widthsj = tj.widths
    Sij = 1.0
    
    for idim in range(ti.numdims):
        xi = ri[idim]
        xj = rj[idim]
        di = pi[idim]
        dj = pj[idim]
        xwi = widthsi[idim]
        xwj = widthsj[idim]
        Sij *= overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)
    
    return Sij

def overlap_nuc_1d(xi, xj, di, dj, xwi, xwj):
    """Compute 1-dimensional nuclear overlaps"""
    
    c1i = (complex(0.0, 1.0))
    deltax = xi - xj
    pdiff = di - dj
    osmwid = 1.0 / (xwi + xwj)
    
    xrarg = osmwid * (xwi*xwj*deltax*deltax + 0.25*pdiff*pdiff)
    if (xrarg < 10.0):
        gmwidth = math.sqrt(xwi*xwj)
        ctemp = (di*xi - dj*xj)
        ctemp = ctemp - osmwid * (xwi*xi + xwj*xj) * pdiff
        cgold = math.sqrt(2.0 * gmwidth * osmwid)
        cgold = cgold * math.exp(-1.0 * xrarg)
        cgold = cgold * cmath.exp(ctemp * c1i)
    else:
        cgold = 0.0
           
    return cgold

def kinetic_nuc_elec(ti, tj, positions_i="positions", positions_j="positions",\
                     momenta_i="momenta", momenta_j="momenta"):
    """Compute the kinetic energy matrix element between two virbonic TBFs"""
    
    if ti.istate == tj.istate:
        Tij = kinetic_nuc(ti, tj, positions_i=positions_i, positions_j=positions_j,\
                          momenta_i=momenta_i, momenta_j=momenta_j)
    else:
        Tij = complex(0.0, 0.0)
    
    return Tij

def kinetic_nuc(ti, tj, positions_i="positions", positions_j="positions",\
                momenta_i="momenta", momenta_j="momenta"):
    """compute the kinetic energy matrix element between two nuclear TBFs"""
    
    ri = eval("ti." + positions_i)
    rj = eval("tj." + positions_j)
    pi = eval("ti." + momenta_i)
    pj = eval("tj." + momenta_j)
        
    widthsi = ti.widths
    widthsj = tj.widths
    
    massesi = ti.masses

    ndim = ti.numdims
    
    S1D = np.zeros(ndim, dtype=np.complex128)
    T1D = np.zeros(ndim, dtype=np.complex128)
    
    for idim in range(ndim):
        xi = ri[idim]
        xj = rj[idim]
        di = pi[idim]
        dj = pj[idim]
        xwi = widthsi[idim]
        xwj = widthsj[idim]
        m = massesi[idim]

        T1D[idim] = 0.5 * kinetic_nuc_1d(xi, xj, di, dj, xwi, xwj) / m
        S1D[idim] = overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

    Tij = 0.0
    for idim in range(ndim):
        Ttmp = T1D[idim]
        #print "T1D[idim] ", T1D[idim], Ttmp
        for jdim in range(ndim):
            if jdim != idim:
                Ttmp *= S1D[jdim]
                #print "S1D[jdim]", S1D[jdim], Ttmp
        Tij += Ttmp
        #print "Tij ", Tij

    return Tij

def kinetic_nuc_1d(xi, xj, di, dj, xwi, xwj):
    """compute 1-dimensional nuclear kinetic energy matrix elements"""
    c1i = (complex(0.0, 1.0))
    psum = di + dj
    deltax = xi - xj
    dkerfac = xwi + 0.25 * psum * psum - xwi * xwi * deltax * deltax
    dkeifac = xwi * deltax * psum
    olap = overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)
    kinetic = (dkerfac + c1i * dkeifac) * olap

    return kinetic

def Sdot_nuc_elec(ti, tj, positions_i="positions", positions_j="positions",\
                  momenta_i="momenta", momenta_j="momenta", forces_j="forces"):
    """Compute the Sdot matrix element between two vibronic TBFs"""
    
    if ti.istate == tj.istate:
        Sdot_ij = Sdot_nuc(ti, tj, positions_i=positions_i, positions_j=positions_j,\
                           momenta_i=momenta_i, momenta_j=momenta_j, forces_j=forces_j)
    else:
        Sdot_ij = complex(0.0, 0.0)
    
    return Sdot_ij

def Sdot_nuc(ti, tj, positions_i="positions", positions_j="positions", momenta_i="momenta",\
             momenta_j="momenta", forces_j="forces"):
    """Compute the Sdot matrix element between two nuclear TBFs"""
    
    c1i = (complex(0.0, 1.0))
    ri = eval("ti." + positions_i)
    rj = eval("tj." + positions_j)
    pi = eval("ti." + momenta_i)
    pj = eval("tj." + momenta_j)
    fj = eval("tj." + forces_j)
        
    widthsi = ti.widths
    widthsj = tj.widths

    massesi = ti.masses

    ndim = ti.numdims
    
    Sij = overlap_nuc(ti, tj, positions_i=positions_i, positions_j=positions_j,\
                      momenta_i=momenta_i, momenta_j=momenta_j)

    deltar = ri - rj
    psum = pi + pj
    pdiff = pi - pj
    o4wj = 0.25 / widthsj
    Cdbydr = widthsj * deltar - (0.5 * c1i) * psum
    Cdbydp = o4wj * pdiff + (0.5 * c1i) * deltar
    Ctemp1 = Cdbydr * pj / massesi + Cdbydp * fj
    Ctemp = np.sum(Ctemp1)    
    Sdot_ij = Ctemp * Sij
    
    return Sdot_ij
