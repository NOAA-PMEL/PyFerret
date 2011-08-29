## Site-dependent definitions included in Makefiles

## Full path name of the directory containing this file (the ferret root directory).
## Do not use $(shell pwd) since this is included in Makefiles in other directories.
DIR_PREFIX	= $(HOME)/pyferret_64dev

## Machine for which to build Ferret
## Use $(HOSTTYPE) to build natively for the machine you are using
BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= i386-linux

## Python 2.x executable to invoke for build and install.
PYTHON_EXE	= python2.6

## Installation directory for built Ferret.  Using the "install"
## Makefile target circumvents the need to create the fer_*.tar.gz
## files just for creating a Ferret installation.
INSTALL_FER_DIR	= $(FER_DIR)

## Installation directory for HDF5 static libraries
## (contains include and lib or lib64 subdirectories)
# HDF5_DIR	= /usr
# HDF5_DIR	= /usr/local
HDF5_DIR	= /usr/local/hdf5_186
# HDF5_DIR	= /usr/local/hdf5_186_64
# HDF5_DIR	= /usr/local/hdf5_187

## Installation directory for NetCDF static libraries
## (contains include and lib or lib64 subdirectories)
# NETCDF4_DIR	= /usr
# NETCDF4_DIR	= /usr/local
NETCDF4_DIR	= /usr/local/netcdf_412
# NETCDF4_DIR	= /usr/local/netcdf_412_64
# NETCDF4_DIR	= /usr/local/netcdf_413

## Installation directory for readline static libraries
## (contains include and lib or lib64 subdirectories)
## If only shared-object library is available, it will
## be used instead.  Version 6.x needed for PyFerret
# READLINE_DIR	= /
# READLINE_DIR	= /usr
READLINE_DIR	= /usr/local

## Installation directory for libz static library
## (contains lib or lib64 subdirectory)
## If only shared-object library is available, it will
## be used instead.  Version 1.2.5 recommended by NetCDF
# LIBZ_DIR	= /
# LIBZ_DIR	= /usr
LIBZ_DIR	= /usr/local

## Java 1.6 jdk home directory - this may be predefined
## from your shell environment.  If JAVA_HOME is defined,
## $(JAVA_HOME)/bin/javac and $(JAVA_HOME)/bin/jar is
## called to build threddsBrowser.jar; otherwise, it just
## uses javac and jar (from the path).
# JAVA_HOME	= /usr/java/latest
# JAVA_HOME	= /usr/lib/jvm/java-1.6.0-sun
# JAVA_HOME	= /usr/lib/jvm/java-6-sun
JAVA_HOME	= /usr/lib/jvm/java-sun

##
