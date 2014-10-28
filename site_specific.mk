## Site-dependent definitions included in Makefiles
## Also verify the values in external_functions/ef_utility/site_specific.mk

## Full path name of the directory containing this file (the ferret root directory).
## Do not use $(shell pwd) since this is included in Makefiles in other directories.
DIR_PREFIX	= $(HOME)/pyferret_dev

## Machine type for which to build Ferret/PyFerret
##   x86_64-linux      for 64-bit RHEL
##   x86_64-linux-gnu  for 64-bit Ubuntu and many "free" Linux systems
##   i386-linux        for 32-bit RHEL
##   i386-linux-gnu    for 32-bit Ubuntu and many "free" Linux systems
##   i386-apple-darwin for MacOS
BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= x86_64-linux-gnu
# BUILDTYPE	= i386-linux
# BUILDTYPE	= i386-linux-gnu
# BUILDTYPE	= i386-apple-darwin
# BUILDTYPE	= intel-mac

## Python 2.x executable to invoke for build and install.
PYTHON_EXE	= python2.6
# PYTHON_EXE	= python2.7
## The assignment of PYTHONINCDIR should not need any modifications
PYTHONINCDIR   := $(shell $(PYTHON_EXE) -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()")

## Installation directory for built Ferret.  Using the "install"
## Makefile target circumvents the need to create the fer_*.tar.gz
## files just for creating a Ferret installation.
INSTALL_FER_DIR	= $(FER_DIR)

## Installation directory for Cairo-1.8.8 static libraries
## (contains include and lib or lib64 subdirectories) for RHEL5.
## Do not give a location on other systems.  For these systems
## the system-wide shared-object Cairo libraries which are also
## used by Qt4 must be used.
CAIRO_DIR	=
# CAIRO_DIR	= /usr/local/cairo_188

## Installation directory for HDF5 static libraries
## (contains include and lib or lib64 subdirectories)
## Do not give a location if linking to netcdf shared-object libraries
# HDF5_DIR	= /usr
# HDF5_DIR	= /usr/local
# HDF5_DIR	= /usr/local/hdf5-1.8.9
HDF5_DIR	= /usr/local/hdf5_189_64
# HDF5_DIR	= 

## Installation directory for NetCDF static or shared object libraries
## (contains include and lib or lib64 subdirectories)
## If HDF5_DIR (above) is empty, the shared-object netcdf libraries will be used.
# NETCDF4_DIR	= /usr
# NETCDF4_DIR	= /usr/local
# NETCDF4_DIR	= /usr/local/netcdf-4.3.1.1
NETCDF4_DIR	= /usr/local/netcdf_4311_64

## Java home directory - this may be predefined
## from your shell environment.  If JAVA_HOME is defined,
## $(JAVA_HOME)/bin/javac and $(JAVA_HOME)/bin/jar is
## called to build threddsBrowser.jar; otherwise, it just
## uses javac and jar (from the path).
# JAVA_HOME	= /usr/java/default
# JAVA_HOME	= /usr/java/latest
# JAVA_HOME	= /usr/lib/jvm/default-java
# JAVA_HOME	= /usr/lib/jvm/java-oracle
# JAVA_HOME	= /usr/lib/jvm/java-sun
JAVA_HOME	= /usr/lib/jvm/java
# JAVA_HOME	= /Library/Java/JavaVirtualMachines/jdk1.7.jdk/Contents/Home

# PyFerret version number - do not change this
PYFERRET_VERSION = 1.1.0

##
