import math
import cmath
from fms.fmsobj import fmsobj
from fms.traj import traj

def overlap_nuc_elec(ti,tj,positions_i="positions",positions_j="positions",momenta_i="momenta",momenta_j="momenta"):
    if ti.get_istate() == tj.get_istate():
        Sij = overlap_nuc(ti,tj,positions_i=positions_i,positions_j=positions_j,momenta_i=momenta_i,momenta_j=momenta_j)
    else:
        Sij = complex(0.0,0.0)
    return Sij

def overlap_nuc(ti,tj,positions_i="positions",positions_j="positions",momenta_i="momenta",momenta_j="momenta"):
    ri = eval("ti.get_" + positions_i + "()")
    rj = eval("tj.get_" + positions_j + "()")
    pi = eval("ti.get_" + momenta_i + "()")
    pj = eval("tj.get_" + momenta_j + "()")
        
    c1i = (complex(0.0,1.0))
    widthsi = ti.get_widths()
    widthsj = tj.get_widths()
    Sij = 1.0
    for idim in range(ti.get_numdims()):
        xi = ri[idim]
        xj = rj[idim]
        di = pi[idim]
        dj = pj[idim]
        xwi = widthsi[idim]
        xwj = widthsj[idim]

        deltax = xi - xj
        pdiff = di - dj
        osmwid = 1.0/(xwi+xwj)

        xrarg = osmwid*(xwi*xwj*deltax*deltax + 0.25*pdiff*pdiff)

        if (xrarg < 10.0):
            gmwidth = math.sqrt(xwi*xwj)
            ctemp = (di*xi - dj*xj)
            ctemp = ctemp - osmwid * (xwi*xi + xwj*xj) * pdiff
            cgold = math.sqrt(2.0 * gmwidth * osmwid)
            cgold = cgold * math.exp(-1.0*xrarg)
            cgold = cgold * cmath.exp(ctemp * c1i)
        else:
            cgold = 0.0

        return cgold
                        
        
