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

## Installation directory for HDF5 static libraries
## (contains include and lib or lib64 subdirectories)
# HDF5_DIR	= /usr
# HDF5_DIR	= /usr/local
# HDF5_DIR	= /usr/local/hdf5_186
HDF5_DIR	= /usr/local/hdf5_187

## Installation directory for NetCDF static libraries
## (contains include and lib or lib64 subdirectories)
# NETCDF4_DIR	= /usr
# NETCDF4_DIR	= /usr/local
# NETCDF4_DIR	= /usr/local/netcdf_412
NETCDF4_DIR	= /usr/local/netcdf_413

## Installation directory for readline static libraries
## (contains include and lib or lib64 subdirectories)
# READLINE6_DIR	= /
# READLINE6_DIR	= /usr
READLINE6_DIR	= /usr/local

## Installation directory for libz static library
## (contains lib or lib64 subdirectory)
# LIBZ125_DIR	= /
# LIBZ125_DIR	= /usr
LIBZ125_DIR	= /usr/local

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
