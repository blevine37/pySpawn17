#from setuptools import setup
from distutils.core import setup

setup(name='pyspawn',
      version='0.0',
      description='Multiple Cloning in Dense Manifolds of States',
      author='Dmitry A. Fedorov, Benjamin G. Levine',
      url='https://github.com/blevine37/pySpawn17/MCDMS',
      packages=['pyspawn', 'pyspawn.classical_integrator', 'pyspawn.potential', 'pyspawn.qm_hamiltonian', 'pyspawn.qm_integrator'],
      #install_requires=['numpy', 'h5py']
      )
