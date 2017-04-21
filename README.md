# PyFerret
The PyFerret Python module from NOAA/PMEL.
This repository is regularly synchronized with PyFerret repository at PMEL
(the pyferret branch of the ferret project in the TMAP SVN repository at PMEL)
using git-svn.

## Jupyter / iPython notebook

The latest ferretmagic module from Patrick Brockmann for using PyFerret 
with the iPython notebook can be obtained using `pip install ferretmagic`, or see
[http://pypi.python.org/pypi/ferretmagic](http://pypi.python.org/pypi/ferretmagic).
Note that this only installs the ferretmagic module for PyFerret;
it does not install PyFerret.

## Anaconda Installation - Linux, OS X, and Windows 10/bash

Download and install [miniconda](http://conda.pydata.org/miniconda.html) for your system. 
Note that Windows 10 bash must use the Linux version! 
The Python version in miniconda does not matter; 
`pyferret` only uses `Python 2.7` and will install it in the `pyferret` environment.

Execute the following command on the terminal to install `pyferret` as well as 
`ferret_datasets` (the default Ferret/PyFerret datasets) into conda:
```shell
conda create -n FERRET -c conda-forge pyferret ferret_datasets --yes
```

To start using `pyferret`, execute the following command:
```shell
source activate FERRET
```

Once you are done working with `pyferret` you can leave this environment, 
if you wish, with the command:
```shell
source deactivate FERRET
```

In the commands above, `FERRET` is the environment name where `pyferret` is installed. 
You can change that to any name you like but we do not recommend installing `pyferret` 
in the root environment of miniconda. 
The main reason is to take advantage of the `activate/deactivate` script that will set 
up all the variables that `pyferret` needs. 
(You can test whether the `pyferret` environment is activated by issuing the command 
`echo $FER_DATA` and see if it returns a directory name.)

## Installation from prebuilt tar.gz file

The following packages are needed to run PyFerret on Ubuntu:
 * `python-numpy`
 * `python-scipy` (optional but highly recommended)
 * `python-pyshp` (or use `pip install pyshp`; for shapefile functions; optional)
 * `libgfortran` (should be installed with python-scipy installation)
 * `default-jre` (or `default-jdk`; for ThreddsBrowser; optional)

The following packages are also needed but should already be installed:
 * `python-qt4` (Python bindings for Qt4)
 * `libcurl3` or `libcurl4-openssl-dev` (for the libcurl.so.4 library)
 * `libpangocairo` (for the pango and pango-cairo text processing libraries)

If you do not already have them, the Ferret standard datasets can be obtained 
from the [FerretDatasets github site](https://github.com/NOAA-PMEL/FerretDatasets)
The contents can be put extracted/cloned to whatever location desired.

Extract the PyFerret tar.gz file in the desired location.
Starting with PyFerret v7, there is only one tar.gz file which 
extracts all its contents to a subdirectory that it creates
(as apposed to Ferret which has separate `fer_environment` and 
`fer_executables` tar.gz files that extract into the current directory).
If desired, at this time you can change the name of this subdirectory 
that was created.

Move into this PyFerret installation directory and run the `bin/Finstall`
script to create the `ferret_paths.sh`, `ferret_paths.csh`, and `pyferret`
scripts.  The value of `FER_DIR`, the Ferret/PyFerret installation directory,
should be this installation directory, which can be specified as `.` (a period) 
which means the current directory.
(If `FER_DIR` is already defined for another Ferret/PyFerret installation,
you will need to tell the script to use a new value.)
For `FER_DSETS`, the Ferret standard datasets, specify the directory 
containing these datasets (which you may have created from the FerretDatasets
github site mentioned above).

To run PyFerret, you first need to set the Ferret environment variables.
This can be done by executing either `. ferret_paths.sh` (for Bourne-type shells; 
e.g., bash) or `source ferret_paths.csh` (for C-type shells; e.g. tcsh).
* Note: the pyferret script has recently been updated to automatically set 
  the Ferret environment variables, if not already defined, using the appropriate 
  `ferret_paths` script.

For more information on using PyFerret, see the Ferret and PyFerret documentation under 
[http://ferret.pmel.noaa.gov/Ferret/documentation/](http://ferret.pmel.noaa.gov/Ferret/documentation/)
Information about the Ferret email users group, and archives of past discussions
from the group (which should be searched prior to sending a question to the email 
users group) can be found at 
[http://ferret.pmel.noaa.gov/Ferret/email-users-group](http://ferret.pmel.noaa.gov/Ferret/email-users-group)


## Building PyFerret from source

To build PyFerret from source code, please see the `Building PyFerret` instructions at
[http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret/build-install/](http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret/build-install/)

