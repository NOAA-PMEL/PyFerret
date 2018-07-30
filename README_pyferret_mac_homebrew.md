# PyFerret on Mac OS X with Homebrew Packages

Information about using or building PyFerret on Mac OS X using
the Homebrew package manager to install required packages.
If you do not have Homebrew installed, see https://brew.sh/
for information on installing and using Homebrew.

Because using the pre-built PyFerret package on Mac OS X typically
requires Homebrew for Python and PyQt, many users have found it
almost as easy and cleaner to just build PyFerret from source using
the Homebrew packages.


## Using pre-built PyFerret on Mac OS X with Homebrew Packages

#### Homebrew Packages

I have always installed the Homebrew Python 3 (`python` or `python3`)
and/or Python 2 (`python@2`) packages and have not attempted to use
the system-provided Python 2.
The Homebrew PyQt5 package requires the Homebrew Python package(s),
so installing PyQt from Homebrew will also install Homebrew Python.
However, if you already have Python 2 or 3 with either PyQt5 or
PyQt4 and with NumPy, then the prebuilt package *should* work,
althought this has not been tested at this time.

Install the `python`, `pyqt`, and `numpy` Homebrew packages.

    brew install python3
    brew install pyqt5
    brew install numpy

While not required, SciPy is highly recommended for its wealth
of scientific modules and for using the statistical functions
provided in PyFerret.

    brew install scipy

If you wish to use the shapefile functions provided in PyFerret,
install the `pyshp` module PyShp https://github.com/GeospatialPython/pyshp/
using `pip2` or `pip3` (part of the Homebrew python@2 and python3 packages,
respectively).

    pip3 install pyshp

#### PyFerret

Download the PyFerret prebuilt Mac OS X tar.gz file from
https://github.com/NOAA-PMEL/PyFerret/releases/
and extract its contents to the desired parent directory/folder
for PyFerret programs.
(Double-clicking on the icon of a tar.gz file will extract
the contents in the same location as the tar.gz file.
Alternatively, the command `tar xzf` followed by the tar.gz filename
will extra the contents to the current working directory.)
Feel free at this time to rename or move this extracted
subdirectory/folder; this is your PyFerret installation directory.

If you do not already have the default Ferret datasets,
also download that "source code" zip or tar.gz file from
https://github.com/NOAA-PMEL/FerretDatasets/releases/
and extract its contents to the desired parent directory/folder
for datasets.
Again, feel free at this time to rename or move this extracted
subdirectory/folder; this is your Ferret datasets directory.

Run the `Finstall` script that is found under the `bin` subdirectory
of the extracted PyFerret installation directory to create the
ferret_paths scripts as well as the pyferret script.
The Finstall script will ask for the PyFerret installation directory
name, the default Ferret datasets directory name, the directory
in which to create the ferret_paths scripts, and the python to use.
These directories can all be specified as relative path names
to the current directory (such as `.` for the current directory)
when you run the Finstall script.

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

Install the `gcc`, `cairo`, `pango`, `python`, `pyqt`, `numpy`, `hdf5`,
and `netcdf` Homebrew packages.

    brew install gcc
    brew install cairo
    brew install pango
    brew install python
    brew install pyqt
    brew install numpy
    brew install hdf5
    brew install netcdf

While not required, SciPy is highly recommended for its wealth
of scientific modules and for using the statistical functions
provided in PyFerret.

    brew install scipy

If you wish to use the shapefile functions provided in PyFerret,
install the `pyshp` module PyShp https://github.com/GeospatialPython/pyshp/
using `pip2` or `pip3` (part of the Homebrew python@2 and python3 packages,
respectively).

    pip3 install pyshp

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
    INSTALL_FER_DIR = /usr/local/PyFerret
    BUILDTYPE = intel-mac
    PYTHON_EXE = python3.7
    GFORTRAN_LIB = $(shell $(FC) --print-file-name=libgfortran.dylib)
    CAIRO_LIBDIR = /usr/local/lib
    PIXMAN_LIBDIR = /usr/local/lib
    PANGO_DIR = /usr/local/lib
    GLIB2_LIBDIR = /usr/local/lib
    HDF5_LIBDIR =
    SZ_LIBDIR =
    NETCDF_LIBDIR = /usr/local/lib

Similarly, copy `external_functions/ef_utility/site_specific.mk.in` to
`external_functions/ef_utility/site_specific.mk` and edit
`external_functions/ef_utility/site_specific.mk`:

    BUILDTYPE = intel-mac
    PYTHON_EXE = python3.7

Build PyFerret and install it in the location given by the value of
`INSTALL_FER_DIR` in the site_specific.mk file:

    make
    make install

At this point you are now where one would be if you had downloaded and
extracted a prebuilt PyFerret tar file to the PyFerret installation
directory.
As described above, run the `Finstall` script under the bin subdirectory
of this PyFerret installation directory to create the pyferret script as
well as the ferret_paths scripts for setting environment variables needed
by PyFerret; then be sure to "source" the appropriate ferret_paths script
as needed.

#### Testing

To run the benchmark tests, the `nco`, `imagemagick`, and `gs`
Homebrew packages are needed.

    brew install nco
    brew install imagemagick
    brew install gs

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
differences are usually from differences in system-provided fonts, and
so are not significant.
Differences in the log, err, and ncdump output files are shown in the
*.diff files, which might be empty, or might show negligable numerical
differences and insignificant differences in system syntax.

