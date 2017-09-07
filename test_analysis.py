import pyspawn

an = pyspawn.fafile("sim.hdf5")

times, N = an.compute_norms(column_filename = "N.dat")

print times
print N

del an

