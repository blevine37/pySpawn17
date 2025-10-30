                                                                                     
                                                                                     
                          .M"""bgd                                               
                         ,MI    "Y                                               
    `7MMpdMAo.`7M'   `MF'`MMb.   `7MMpdMAo.  ,6"Yb.`7M'    ,A    `MF'`7MMpMMMb.  
      MM   `Wb  VA   ,V    `YMMNq. MM   `Wb 8)   MM  VA   ,VAA   ,V    MM    MM  
      MM    M8   VA ,V   .     `MM MM    M8  ,pm9MM   VA ,V  VA ,V     MM    MM  
      MM   ,AP    VVV    Mb     dM MM   ,AP 8M   MM    VVV    VVV      MM    MM  
      MMbmmd'     ,V     P"Ybmmd"  MMbmmd'  `Moo9^Yo.   W      W     .JMML  JMML.
      MM         ,V                MM                                            
    .JMML.    OOb"               .JMML.



pySpawn17
=========

version 1.0

created by
Benjamin G. Levine
Michigan State University

A trim but extensible full multiple spawning software package, written in python and distributed under the MIT License.


Citation
========

If you use this code, we ask that you cite the paper at the follows DOI:

10.1021/acs.jctc.0c00575

License
=======

See LICENSE file


Features
========

This is an ab initio multiple spawning code written in python.  It is designed to be rather minimalistic, but easily extensible.  Right now, it has the following features:

-  Runs in the adiabatic representation with derivative couplings computed via NPI.
-  Interface to (a development version of) TeraChem via the tcpb interface. 
-  SSAIMS — Stochastic-Selection AIMS (optional per-run)
-  An analysis module for processing of simulation data.

This code is currently under development.  Example jobs are provided.  Documentation is present, but a work in progress.  DGAS is being implemented.  A release is forthcoming after these issues are addressed and the code is more rigorously tested.

Interfaces
==========

At present only an interface for (a development version of) TeraChem is provided, but pySpawn is designed to be readily interfaced with other codes.

Contact
=======

pySpawn is developed and maintained primarily by Benjamin G. Levine, levine@chemistry.msu.edu.





