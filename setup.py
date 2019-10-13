# from setuptools import setup
from distutils.core import setup

setup(name='pyspawn',
      version='0.0',
      description='Full Multiple Spawning in Python',
      author='Benjamin G. Levine',
      url='https://github.com/blevine37/pySpawn17',
      packages=['pyspawn', 'pyspawn.classical_integrator', 'pyspawn.potential', 'pyspawn.qm_hamiltonian',
                'pyspawn.qm_integrator'], requires=['numpy', 'h5py']
      # install_requires=['numpy', 'h5py']
      )
