from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'A basic package with handy function for lab work, made for Jupyter Notebooks'
LONG_DESCRIPTION = 'A package with utility functions to make using python in the lab faster, specifically for the Complutense University of Madrid Physics Bachelor.'

# Setting up
setup(
    name="labhelper",
    version=VERSION,
    author="Batres (Javier Batres)",
    author_email="<javibatresdc@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scipy', 'pandas', 'numpy', 'sympy', 'IPython'],
    keywords=['python', 'data analysis', 'lab', 'data', 'symbolic'],
    classifiers=[
        "Development Status :: Development",
        "Intended Audience :: Students",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)