from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['mppi_tf'],
    package_dir={'': 'scripts/mppi_tf/scripts'}
)
setup(**d)