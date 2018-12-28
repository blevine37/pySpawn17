import os.path

def print_splash():
    print " "
    print "You are about to propagate a molecular wave function with..."
    print " "
    print "                          .M\"\"\"bgd"
    print "                         ,MI    \"Y"
    print "    `7MMpdMAo.`7M'   `MF'`MMb.   `7MMpdMAo.  ,6\"Yb.`7M'    ,A    `MF'`7MMpMMMb."
    print "      MM   `Wb  VA   ,V    `YMMNq. MM   `Wb 8)   MM  VA   ,VAA   ,V    MM    MM"
    print "      MM    M8   VA ,V   .     `MM MM    M8  ,pm9MM   VA ,V  VA ,V     MM    MM"
    print "      MM   ,AP    VVV    Mb     dM MM   ,AP 8M   MM    VVV    VVV      MM    MM"
    print "      MMbmmd'     ,V     P\"Ybmmd\"  MMbmmd'  `Moo9^Yo.   W      W     .JMML  JMML."
    print "      MM         ,V                MM"
    print "    .JMML.    OOb\"               .JMML."
    print " "
    print "pySpawn is a nonadiabatic molecular dynamics software package written by Benjamin G. Levine"
    print " "

def check_files():
    print " Checking for files from previous run"
    if os.path.isfile("working.hdf5"):
        print "! working.hdf5 is present.  Are you sure you want to continue?  Exiting"
        quit()
    if os.path.isfile("sim.hdf5"):
        print "! sim.hdf5 is present.  Are you sure you want to continue?  Exiting"
        quit()
