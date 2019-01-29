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

While the `ferret.pmel.noaa.gov` site is offline, please use the following 
instructions for build PyFerret from the GitHub source files found at this site.
Please note that these are general instructions that are not fully verified; names 
of installation packages may vary slightly for you particular operating system.
In particular, some systems have special development (`-dev`) packages that provide 
the include files, and shared-object libraries without a numeric extension, that are 
needed for compiling and linking the PyFerret code.
On other systems, these include and library files are part of the standard package.
These instructions assume your package manager provides recent versions of HDF5 and NetCDF.

#### Packages from the package manager

If not already installed on your system, install the following packages using the package 
manager for your operating system, or a command-line package installation program such as 
`yum` or `apt-get` (which needs to be run as a system administator - as "root" - or using 
the `sudo` privilege escalation program):  
- `gfortran`, or `gcc` (Gnu Compiler Collection) on some systems - for the gfortran compiler and library  
- `libcairo`, `libcairo-dev`, `cairo`, or `cairo-dev` - for the cairographics library and include files  
- `libpango`, `libpango-dev`, `pango`, or `pango-dev` - for the pango and pango-cairo library and include files  
- `numpy`, or `python-numpy` - the NumPy python package as well as include and library files  
- `pyqt`, `python-qt5`, `python-qt4`, `PyQt5`, or `PyQt4` - for either PyQt5 or PyQt4  
- `netcdf` - for NetCDF 4.x include and library files  

The NetCDF package should add the HDF5 packages as a dependency.
Some package manager programs (such as Homebrew) have their own version of Python separate 
from the operating system; if so, the NumPy and PyQt packages should add the python package(s) 
as dependencies.

You may also want, if not already installed:  
- `git` - to use "git" commands to download the source code; highly recommended  
- `scipy`, or `python-scipy` - for statisticaly functions in PyFerret; highly recommended  
- `pyshp`, or `python-pyshp` - for shapefile functions in PyFerret  

Note that `pyshp` is pure-python code and can also be installed using `pip2`
(part of python2.x) or `pip3` (part of python3.x). 

#### PyFerret source code

The green `Clone or download` button at the top of the PyFerret GitHub 
`Code` page/tab gives you options for obtaining the latest PyFerret source code.
You can get a copy of the latest source as a zip file, but a better option, 
if you can, is to use "git" commands to clone the repository (the source code, 
plus history and version control of the source code) to your local system.

The git comands to clone the PyFerret repository look something like the following
(the local copy of the repository will be put into `$HOME/git/PyFerret`):  
```shell
git clone https://github.com/NOAA-PMEL/PyFerret.git $HOME/git/PyFerret  
cd $HOME/git/PyFerret  
```
Working with a cloned repository allows you to quickly and easily update
and rebuild PyFerret when updates appear that you wish to use.
Executing the command:  
```shell
git pull  
```
when in the PyFerret cloned repository will download any changes to your local copy.

#### Configure and build

In the PyFerret source directory, copy the `site_specific.mk.in` configuration
template file to `site_specific.mk` and edit this `site_specific.mk`
configuration file appropriately for your system; for example:  
```shell
DIR_PREFIX = $(HOME)/git/PyFerret  
INSTALL_FER_DIR = /usr/local/PyFerret  
BUILDTYPE = x86_64-linux  
PYTHON_EXE = python2.7  
GFORTRAN_LIB = $(shell $(FC) --print-file-name=libgfortran.a)  
CAIRO_LIBDIR =   
PIXMAN_LIBDIR =   
PANGO_DIR =   
GLIB2_LIBDIR =   
HDF5_LIBDIR =  
SZ_LIBDIR =  
NETCDF_LIBDIR = /usr/lib64  
```
Information about each of these values, as well as suggested values to assign,
are included as comments (lines starting with a `#`) in the `site_specific.mk` file.

Similarly, copy `external_functions/ef_utility/site_specific.mk.in` to
`external_functions/ef_utility/site_specific.mk` and edit
`external_functions/ef_utility/site_specific.mk`:  
```shell
BUILDTYPE = x86_64-linux  
PYTHON_EXE = python2.7  
```

If you have previously built PyFerret (successfully or not) from this source directory 
or repository, run the command:  
```shell
make clean  
```
to make sure you remove all previously generated files.
Then run the command:  
```shell
make
```
to build PyFerret.
This build will take a bit of time (minutes) and will generate a lot of output,
so you may wish to redirect output and run this command in the background.

When the build has successfully completed, install PyFerret in the location given 
by the value of `INSTALL_FER_DIR` in the site_specific.mk file by running the following
command.  (If the installation directory exists and is not empty, you should 
move or remove any contents of that directory to ensure a clean installation.)  
```shell
make install
```
You may need to be logged in as a system administrator (as "root") or use the "sudo" 
privilege escalation command (thus, `sudo make install`), to install PyFerret 
system-wide (such as installing in `/usr/local/PyFerret` as in the example 
`site_specific.mk` file given above.

#### Standard Ferret/PyFerret datasets

If you do not have the standard Ferret/PyFerret datasets, they can be downloaded from
[the FerretDatasets GitHub repository](https://github.com/NOAA-PMEL/FerretDatasets)
either as a zip file download or as a git cloned repository (similar to obtaining the
PyFerret source).
If you already have a copy of these datasets on your system, these datasets can be 
shared between Ferret and PyFerret, including different versions of these programs.
You can also add any of your own datasets that might be frequently used.
These datasets will be needed as part of the following PyFerret configuration.

#### (Py)Ferret configuration

Change to the PyFerret installation directory (the value of `INSTALL_FER_DIR`) 
created above and run the script:  
```shell
bin/Finstall
```
to create the `ferret_paths.sh`, `ferret_paths.csh`, and `pyferret` scripts.  
The value of `FER_DIR`, the Ferret/PyFerret installation directory, should be this 
installation directory, which can be specified as `.` (a period) which means the 
current directory.
The value of FER_DSETS should be the directory containing the standard 
Ferret/PyFerret dataset mentioned above.

Before running PyFerret for the first time in a new terminal window (shell), 
you will need to set the Ferret/PyFerret environment variables using the 
appropriate ferret_paths script:  
```shell
. /my/path/to/ferret_paths.sh
```
(a period, a space, and the path to the ferret_paths.sh script) for Bourne-type shells (such as bash), or 
```shell
source /my/path/to/ferret_paths.csh
```
for C-type shells (such as tcsh).
(The `source` command is also supported by the `bash` shell.)

