# PyFerret on Mac OS X with Homebrew Packages 
Information about using or building PyFerret on Mac OS X using 
the Homebrew package manager to install required packages. 
If you do not have Homebrew installed, see https://brew.sh/
for information on installing and using Homebrew. 

## Using PyFerret on Mac OS X with Homebrew Packages

#### Homebrew Packages

I have always installed the Homebrew Python 3.6 (`python`) and/or 
Python 2.7 (`python@2`) packages and have not attempted to use the 
system-provided python (2.7). 
You will need to install the Homebrew package for PyQt5 (`pyqt` or 
`pyqt5`, which require Python from Homebrew). 

    brew install python
    brew install pyqt

Then, using `pip` (part of the Homebrew Python packages), install 
the `numpy`, `scipy`, and `pyshp` packages. 
NumPy http://www.numpy.org/ is required to use PyFerret; 
SciPy https://www.scipy.org/ and 
PyShp https://github.com/GeospatialPython/pyshp/ 
are optional but are highly recommended as they enable 
statistical and shapefile functions in PyFerret.

    pip install numpy
    pip install scipy
    pip install pyshp

#### PyFerret

Download the MacOSX prebuilt tar.gz file for PyFerret from 
https://github.com/NOAA-PMEL/PyFerret/releases/ and extract its 
contents to the desired parent directory for PyFerret programs.  

If you do not already have the default Ferret datasets, also download 
that zip or tar.gz file from 
https://github.com/NOAA-PMEL/FerretDatasets/releases/ 
and extract its contents to the desired parent directory for datasets.

Run the `Finstall` script that is found under the `bin` subdirectory 
of the extracted PyFerret installation directory. 
This script will ask for the PyFerret installation directory name, 
the default Ferret datasets directory name, the directory in which 
to create the ferret_paths scripts, and the python to use. 
These directories can all be specified as relative path names to the 
current directory when you run the Finstall script.

The ferret_paths scripts are used to assign environment variables 
required to run PyFerret.
Thus, you must "source" the appropriate ferret_paths script 
(`source ferret_paths.csh` for C-shell users, or `. ferret_paths.sh` 
for Bourne-shell users) prior to running PyFerret for the first 
time in a command window.
If desired, this can be done in a shell startup script such as 
`$HOME/.cshrc` or `$HOME/.bashrc`

## Building PyFerret on Mac OS X with Homebrew Packages.

#### Homebrew Packages

Make sure the bin subdirectory of the Homebrew package installation 
directory (`/usr/local/bin` is the default) is in your PATH environment
variable: 

    export PATH="/usr/local/bin:$PATH"

Install the `gcc`, `cairo`, `pango`, `python`, `pyqt`, `hdf5`, and 
`netcdf` Homebrew packages.

    brew install gcc
    brew install cairo
    brew install pango
    brew install python
    brew install pyqt
    brew install hdf5
    brew install netcdf

Then, using `pip` (part of the Homebrew Python packages), install the 
`numpy`, `scipy`, and `pyshp` packages.  (The `scipy` and `pyshp` 
packages are optional but highly recommended.) 

    pip install numpy
    pip install scipy
    pip install pyshp

#### PyFerret

Obtain the PyFerret source code from GitHub. 
You could just download the source code zip or tar.gz file for a PyFerret 
release from https://github.com/NOAA-PMEL/PyFerret/releases/ and extract 
it to the desired parent directory for building PyFerret. 
Or, if you are familiar with `git`, you can clone the PyFerret repository 
to build from the latest development code:

    git clone https://github.com/NOAA-PMEL/PyFerret.git $HOME/git/PyFerret
    cd $HOME/git/PyFerret

Working with a cloned repository allows you to quickly and easily update 
and rebuild PyFerret when updates appear that you wish to use.
(If you wish to help with PyFerret development, fork the NOAA-PMEL/PyFerret 
repository to a repository under your own GitHub account and then clone your 
copy of the repository to your local system.)

In the PyFerret source directory, copy the `site_specific.mk.in` configuration 
template file to `site_specific.mk` and edit this `site_specific.mk` 
configuration file appropriately for your system; for example:

    DIR_PREFIX = $(HOME)/git/PyFerret
    BUILDTYPE = intel-mac
    PYTHON_EXE = python3.6
    INSTALL_FER_DIR = /usr/local/PyFerret
    CAIRO_DIR = /usr/local
    PIXMAN_DIR = /usr/local
    HDF5_DIR = /usr/local
    COMPRESS_LIB = sz
    NETCDF4_DIR = /usr/local

Similarly, copy `external_functions/ef_utility/site_specific.mk.in` to 
`external_functions/ef_utility/site_specific.mk` and edit 
`external_functions/ef_utility/site_specific.mk`:

    BUILDTYPE = intel-mac
    PYTHON_EXE = python3.6
    INSTALL_FER_DIR = /usr/local/PyFerret

Build PyFerret and install it in the location given by `INSTALL_FER_DIR`:

    make
    make install

At this point you are now where one would be if you had downloaded and 
extracted a prebuilt PyFerret tar file to the PyFerret installation 
directory.
Run the `Finstall` script under the bin subdirectory of the PyFerret 
installation directory to create the ferret_paths scripts for setting 
environment variables needed by PyFerret.

#### Testing

To run the benchmark tests, the `nco` and `imagemagick` Homebrew 
packages are needed.

    brew install nco
    brew install imagemagick

Make sure the environment variables need by PyFerret are assigned 
appropriately using the appropriate ferret_paths script, and that 
the command `which pyferret` returns the path to the version of 
PyFerret that you wish to test.

Change to the `bench` subdirectory of the PyFerret source directory, 
and run `make run_tests`.
These tests will generate image files as well as `all_..._log` 
(normal output), `all_..._err` (error output), and `all_..._ncdump` 
(human-readable versions of the created NetCDF files) output files.
These files are then compared to similar files under the `test_results` 
subdirectory to create differences files.

Any actual differences in the PNG and PDF images are show in dark red 
(overlaid on a faded original image) in any *_diff.png images created. 
There might not be any difference image files, but if there are, the 
differences are usually from differnces in system-provided fonts and 
so are not significant.
Differences in the log, err, and ncdump output files are shown in the 
*.diff files, which might be empty, or might show negligable numerical 
differences and insignificant differences in system syntax.

