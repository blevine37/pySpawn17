import os.path

def print_splash():
    print " "
    print "You are about to propagate a molecular wave function with MCDMS"
    print "\n----------------------------------------------"
    print "Multiple Cloning in Dense Manifolds of states"
    print "Nonadiabatic molecular dynamics software package" +\
          "written by\nBenjamin G. Levine, Dmitry A. Fedorov"
    print "------------------------------------------------"

    print " "

def check_files():
    print " Checking for files from previous run"
    if os.path.isfile("working.hdf5"):
        print "! working.hdf5 is present.  Are you sure you want to continue?  Exiting"
        quit()
    if os.path.isfile("sim.hdf5"):
        print "! sim.hdf5 is present.  Are you sure you want to continue?  Exiting"
        quit()
