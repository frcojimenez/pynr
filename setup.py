
from setuptools import Extension, setup, Command
import setuptools


with open("pynr/_version.py", "r") as fh:
    vstr = fh.read().strip()
    vstr = vstr.split('=')[1].strip()
    version = vstr.replace("'", "").replace('"', "")
    
setuptools.setup (
    name = 'pynr',
    version = version,
    description = 'Code to load the data from NR catalogues',
    long_description = open('README.md').read(),
    author = 'Xisco Jimenez Forteza',
    author_email = 'francisco.jimenez.forteza@aei.mpg.de',
    url = 'http://www.pycbc.org/',
    download_url = 'https://github.com/frcojimenez',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'h5py','glob2','jsons','sympy','lalsuite','sxs','romspline'],
    keywords = ['nr data', 'signal processing', 'gravitational waves'],
    py_modules = ['pynr','util'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    python_requires='>=3.7.3',
)