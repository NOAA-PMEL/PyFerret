# Ferret on Mac OS X with Homebrew Packages
Information about building Ferret on Mac OS X using the Homebrew package 
manager to install required packages.
If you do not have Homebrew installed, see https://brew.sh/ for information 
on installing and using Homebrew.

## Building Ferret on Mac OS X with Homebrew Packages.

#### Homebrew Packages

Make sure you have the `Xcode` application installed (available for free from 
the Apple `App Store` application), and have run `xcode-select --install` to 
add the command-line developer tools.

Also make sure the bin subdirectory of the Homebrew package installation 
directory (`/usr/local/bin` is the default) is in your PATH environment 
variable, for example:

    export PATH="/usr/local/bin:$PATH"

Install the `gcc`, `readline`, `hdf5`, and `netcdf` Homebrew packages.

    brew install gcc
    brew install readline
    brew install hdf5
    brew install netcdf

#### Ferret

Obtain the Ferret source code from GitHub.
You could just download the source code zip or tar.gz file for a Ferret release 
from https://github.com/NOAA-PMEL/Ferret/releases/ and extract it to the desired 
parent directory for building Ferret.
Or, if you are familiar with `git`, you can clone the Ferret repository to build 
from the latest development code:

    git clone https://github.com/NOAA-PMEL/Ferret.git $HOME/git/Ferret
    cd $HOME/git/Ferret

Working with a cloned repository allows you to quickly and easily update and 
rebuild Ferret when updates appear that you wish to use.
(If you wish to help with Ferret development, fork the NOAA-PMEL/Ferret 
repository to a repository under your own GitHub account and then clone your copy 
of the repository to your local system.)

In the Ferret source directory, copy the `site_specific.mk.in` configuration 
template file to `site_specific.mk` and edit this `site_specific.mk` configuration 
file appropriately for your system; for example:

    DIR_PREFIX = $(HOME)/git/Ferret (wherever you cloned or copied the Ferret source)
    BUILDTYPE = intel-mac
    INSTALL_FER_DIR = /usr/local/Ferret-7.4 (wherever you want Ferret installed)
    HDF5_LIBDIR =
    SZ_LIBDIR =
    NETCDF_LIBDIR = /usr/local/lib (wherever Homebrew put its netcdf library)
    READLINE_LIBDIR = /usr/local/Cellar/readline/7.0.3_1/lib (wherever Homebrew put its hidden readline library)

Similarly, copy `external_functions/ef_utility/site_specific.mk.in` to
`external_functions/ef_utility/site_specific.mk` and edit
`external_functions/ef_utility/site_specific.mk`:

    BUILDTYPE = intel-mac

Build Ferret and install it in the location given by `INSTALL_FER_DIR`:

    make
    make install

At this point you are now where one would be if you had downloaded and extracted 
a prebuilt Ferret tar file to the Ferret installation directory.
Run the `Finstall` script under the bin subdirectory of the Ferret installation 
directory to create the ferret_paths scripts for setting environment variables 
needed by Ferret.

#### Testing

To run the benchmark tests, the `nco`, `imagemagick`, and `gs` Homebrew packages 
are needed.

    brew install nco
    brew install imagemagick
    brew install gs

Make sure the environment variables need by Ferret are assigned appropriately 
using the appropriate ferret_paths script, and that the command `which pyferret` 
returns the path to the version of Ferret that you wish to test.

Change to the `bench` subdirectory of the Ferret source directory, and run 
`make run_tests`.
These tests will generate image files as well as `all_..._log` (normal output), 
`all_..._err` (error output), and `all_..._ncdump` (human-readable versions of 
the created NetCDF files) output files.
These files are then compared to similar files under the `test_results` 
subdirectory to create differences files.

Any actual differences in the GIF images are show in bright red (overlaid on 
a faded original image) in any *_diff.png images created.
There usually are difference image files, but rarely do they show any actual
differences (so just the faded original image with no bright red).
Differences in the log, err, and ncdump output files are shown in the *.diff 
files, which might be empty, or might show negligable numerical differences and 
insignificant differences in system syntax.

