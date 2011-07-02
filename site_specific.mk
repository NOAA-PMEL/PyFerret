## Site-dependent definitions included in Makefiles

## Full path name of the directory containing this file (the ferret root directory).
## Do not use $(shell pwd) since this is included in Makefiles in other directories.
DIR_PREFIX	:= $(HOME)/pyferret_64dev

## Python 2.x executable to invoke for build and install.
PYTHON_EXE	:= python2.6

## Java 1.6 jdk home directory
## $(JAVA_HOME)/bin/javac is called to build threddsBrowser.jar
JAVA_HOME	:= /usr/java/latest
# JAVA_HOME	:= /usr/lib/jvm/java-1.6.0-sun
# JAVA_HOME	:= /usr/lib/jvm/java-6-sun

## Installation directory for HDF5 (contains include and lib subdirectories)
HDF5_DIR	:= /usr/local/hdf5_186

## Installation directory for NetCDF (contains include and lib subdirectories)
NETCDF_DIR	:= /usr/local/netcdf_412

## Installation directory for readline v6.x (contains include and lib subdirectories)
READLINE_DIR	:= /usr/local/readline_62

##
