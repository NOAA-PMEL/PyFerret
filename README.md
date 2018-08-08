# PyFerret
The PyFerret Python module from NOAA/PMEL.  
This repository is regularly synchronized with PyFerret repository at PMEL 
(the pyferret branch of the ferret project in the subversion repository at 
PMEL) using git-svn.

#### Legal Disclaimer
*This repository is a software product and is not official communication 
of the National Oceanic and Atmospheric Administration (NOAA), or the 
United States Department of Commerce (DOC).  All NOAA GitHub project 
code is provided on an 'as is' basis and the user assumes responsibility 
for its use.  Any claims against the DOC or DOC bureaus stemming from 
the use of this GitHub project will be governed by all applicable Federal 
law.  Any reference to specific commercial products, processes, or services 
by service mark, trademark, manufacturer, or otherwise, does not constitute 
or imply their endorsement, recommendation, or favoring by the DOC. 
The DOC seal and logo, or the seal and logo of a DOC bureau, shall not 
be used in any manner to imply endorsement of any commercial product 
or activity by the DOC or the United States Government.*

## Ferret/PyFerret Documentation

For more information on using PyFerret, see the Ferret and PyFerret documentation under 
[http://ferret.pmel.noaa.gov/Ferret/](http://ferret.pmel.noaa.gov/Ferret/)

Information about the Ferret email users group, and archives of past discussions
from the group (which should be searched prior to sending a question to the email 
users group) can be found at 
[http://ferret.pmel.noaa.gov/Ferret/email-users-group](http://ferret.pmel.noaa.gov/Ferret/email-users-group)

## If you build PyFerret from these source files, please note:
The `site_specific.mk` and `external_functions/ef_utilites/site_specific.mk` 
files in the repository have been renamed with a `.in` appended to the name. 
You must copy these files with the `.in` extensions to create files with the 
`site_specific.mk` name and edit the contents to configure these for your 
system.  The `site_specific.mk` files will be ignored by git (the name was 
added to `.gitignore`) so your customized configuration files will not be 
added to your repository if you have cloned this repository. 

#### The `site_specific.mk.in` configuration template file changed April, 2018.

Library directories are now specified instead of installation directories,
with the makefile environment names changing from `..._DIR` to `..._LIBDIR`.
The `include` directories, if needed, are assumed to be sibling directories
to these library directories.
You may wish to create a new `site_specific.mk` configuration file 
from this updated configuration template file.

## Jupyter / iPython notebook

The latest ferretmagic module from Patrick Brockmann for using PyFerret 
with the iPython notebook can be obtained using `pip install ferretmagic`, or see
[http://pypi.python.org/pypi/ferretmagic](http://pypi.python.org/pypi/ferretmagic).
Note that this only installs the ferretmagic module for PyFerret;
it does not install PyFerret.

## Anaconda Installation - Linux, OS X, and Windows 10/bash

Download and install [miniconda](http://conda.pydata.org/miniconda.html) for your system.
Note that Windows 10 bash must use the Linux version!
Either python version of miniconda is fine; `pyferret` works with ether python 2.x or python 3.x

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

You will need to have the following packages installed using your software manager
application, or using a command-line package installation program such as `yum` or
`apt-get` (which needs to be run as the root user or using the `sudo` privilege
escalation program.)

Required packages that may not already be installed:
- `numpy` or `python-numpy` (NumPy)
- `libgfortran` (Fortran library; if you install SciPy, it will be installed)
- `PyQt4` or `python-qt4` (Python binding for Qt4; may already be installed)

Highly recommended but optional packages:
- `scipy` or `python-scipy` (SciPy)
- `pyshp` or `python-pyshp` (PyShp for shapefile functions)

You may also wish to install the `netcdf` and `nco` packages to provide some useful
programs for working with NetCDF files (such as `ncdump` and `ncattted` which are used
in the benchmark tests).

If you do not have the Ferret/PyFerret standard datasets, they can be obtained from the
[NOAA-PMEL/FerretDatasets](https://github.com/NOAA-PMEL/FerretDatasets) GitHub repo.
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
For `FER_DSETS`, the Ferret/PyFerret standard datasets, specify the directory
containing these datasets (which you may have created from the FerretDatasets
github site mentioned above).

To run PyFerret, you first need to set the PyFerret environment variables.
This can be done by executing either `. ferret_paths.sh` (for Bourne-type shells;
e.g., bash) or `source ferret_paths.csh` (for C-type shells; e.g. tcsh).
- Note: the pyferret script has recently been updated to automatically set
  the Ferret environment variables, if not already defined, using the appropriate
  `ferret_paths` script.

## Building PyFerret from source

To build PyFerret from source code, please see the `Building PyFerret` instructions at
[http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret/build-install/](http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret/build-install/)
Please note that the `site_specific.mk` and `external_functions/ef_utilites/site_specific.mk`
files in the repository have been renamed with a `.in` appended to the name.
You must copy these files with the `.in` extensions to create files with the
`site_specific.mk` name and edit the contents to configure these for your
system.  The `site_specific.mk` files will be ignored by git (the name was
added to `.gitignore`) so your customized configuration files will not be
added to your repository if you have cloned this repository.

