Instructions for build PyFerret on Mac OSX using homebrew.
If you do not have homebrew install, run the following at a command prompt (terminal window).
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
See http://brew.sh/ for more information on using homebrew.

You will want to have the bin subdirectory of the homebrew package installation directory 
(/usr/local/bin is the default) at the start of your path :

export PATH="/usr/local/bin:$PATH"

Install the gcc package using homebrew in order to get a gfortran compiler :

brew install gcc

Assign some environment variables specifying the compilers from the gcc install :
( for C-shell users, use 'setenv SYM "VAL"' instead of 'export SYM="VAL"' )

export CC="/usr/local/bin/gcc-5"
export FC="/usr/local/bin/gfortran-5"
export F77="/usr/local/bin/gfortran-5"
export CXX="/usr/local/bin/g++-5"

Install some more packages using homebrew :

brew install cairo
brew install pango
brew install python
brew install pyqt

Install packages using pip (which was installed as part of the python package) :

pip install numpy
pip install scipy
pip install pyshp

We typically build and install the latest HDF5 and NetCDF4 static libraries
for linking into Ferret and PyFerret.  However, if you can find packages
that install these for you that you are happy using, skip down to 
"Get the PyFerret source code from github"

Assign some compiler flags environment variables for compiling HDF5 :

export CPPFLAGS="-fPIC -I/usr/local/include"
export CFLAGS="-fPIC -I/usr/local/include"
export FCFLAGS="-fPIC -I/usr/local/include"
export LDFLAGS="-fPIC -L/usr/local/lib"

Download, then build and install the latest release of HDF5 from http://www.hdfgroup.org/ftp/HDF5/current/src/ :

tar xzf Downloads/hdf5-1.8.16.tar.gz
cd hdf5-1.8.16
./configure --prefix=/usr/local/hdf5-1.8.16 --disable-shared
make
make check
make install
cd ..
rm -fr hdf5-1.8.16

Assign some compiler flags enviroment variables for compiling NetCDF C :

export CPPFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include"
export CFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include"
export FCFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include"
export LDFLAGS="-fPIC -L/usr/local/lib -L/usr/local/hdf5-1.8.16/lib"

Download, then build and install the latest release of NetCDF C library from http://www.unidata.ucar.edu/downloads/netcdf/ :

tar xzf Downloads/netcdf-4.4.0.tar.gz
cd netcdf-4.4.0
./configure --prefix=/usr/local/netcdf-4.4.0 --disable-shared --disable-dap-remote-tests
make
make check
make install
cd ..
rm -fr netcdf-4.4.0

Assign some compiler flags enviroment variables for compiling NetCDF Fortran :

export CPPFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include -I/usr/local/netcdf-4.4.0/include"
export CFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include -I/usr/local/netcdf-4.4.0/include"
export FCFLAGS="-fPIC -I/usr/local/include -I/usr/local/hdf5-1.8.16/include -I/usr/local/netcdf-4.4.0/include"
export LDFLAGS="-fPIC -L/usr/local/lib -L/usr/local/hdf5-1.8.16/lib -L/usr/local/netcdf-4.4.0/lib"
export LIBS="-lnetcdf -lhdf5_hl -lhdf5 -ldl -lm -lz -lcurl"

Download, then build and install the latest release of NetCDF Fortran library from http://www.unidata.ucar.edu/downloads/netcdf/ :

tar xzf Downloads/netcdff-4.4.3.tar.gz
cd netcdff-4.4.3
./configure --prefix=/usr/local/netcdf-4.4.0 --disable-shared --disable-f03
make
make check
make install
cd ..
rm -fr netcdff-4.4.3

Clear compiler flags environment variables before building PyFerret :
( for C-shell users, use 'unsetenv SYM' instead of 'export SYM=""' )

export CPPFLAGS=""
export CFLAGS=""
export FCFLAGS=""
export LDFLAGS=""
export LIBS=""

Get the PyFerret source code from github :

git clone https://github.com/NOAA-PMEL/PyFerret.git pyferret_src
cd pyferret_src

Or if you wish to help with PyFerret development, fork NOAA-PMEL/PyFerret under your own github account 
and then clone the repo under your account to your local system.

Edit site_specific.mk appropriately for your system :

DIR_PREFIX = /your/path/to/pyferret_src
BUILDTYPE = intel-mac
PYTHON_EXE = python2.7
INSTALL_FER_DIR = /your/path/to/where/to/install/PyFerret
CAIRO_DIR = /usr/local
PIXMAN_DIR = /usr/local
HDF5_DIR = /usr/local/hdf5-1.8.16
NETCDF4_DIR = /usr/local/netcdf-4.4.0
JAVA_HOME = /Library/Java/JavaVirtualMachines/jdk1.8.0_60.jdk/Contents/Home

(JAVA_HOME will need to match whatever version number is on your system.)

Edit external_functions/ef_utility/site_specific.mk similarily :

BUILDTYPE = intel-mac
PYTHON_EXE = python2.7
INSTALL_FER_DIR = /your/path/to/where/to/install/PyFerret

Build PyFerret and install in the location given by INSTALL_FER_DIR :

make
make install
cd ..

At this point you are now where one would be if you had downloaded and 
extracted a prebuilt PyFerret tar file to the PyFerret installation 
directory.

Run the Finstall script under the PyFerret installation directory 
and assign the ferret_paths environment variables for running PyFerret :
( for C-shell users, use 'source path/to/ferret_paths.csh' instead of '. path/to/ferret_paths.sh' )

cd /your/path/to/where/to/install/PyFerret
bin/Finstall
. path/to/ferret_paths.sh

The benchmark tests are found in the bench subdirectory of the pyferret_src directory.
You can use RUN_TESTS.sh to run through the suite of tests and examine the results
in the all_... files that are generated as well as the PDF files that are created,
and compare them to the files in the test_results subdirectory in bench.

cd /your/path/to/pyferret_src
cd bench
./RUN_TESTS.sh `which pyferret` "$PYFER_EXTERNAL_FUNCTIONS" <Any comment you wish to make>

