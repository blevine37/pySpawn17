import pyspawn
import matplotlib
import matplotlib.pyplot as plt

an = pyspawn.fafile("sim.hdf5")

times, N = an.compute_norms(column_filename = "N.dat")

print times
print N[0,:]

del an

plt.plot(times,N[0,:],"ro",times,N[1,:],"bo",markeredgewidth=0.0)
plt.xlabel('Time')
plt.ylabel('Population')
plt.savefig('N.png')

