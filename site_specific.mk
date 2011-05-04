## Site-dependent definitions included in Makefiles

## Full path name of the directory containing this file (the ferret root directory).
## Do not use $(shell pwd) since this is included in Makefiles in other directories.
# DIR_PREFIX	:= $(HOME)/pyferret_32dev
DIR_PREFIX	:= $(HOME)/pyferret_64dev

## Python 2.x executable to invoke for build and install.
# PYTHON_EXE	:= python2.4
PYTHON_EXE	:= python2.6

## Flags for specifying the installation directory for "$(PYTHON_EXE) setup.py install"
## The following is for the user-specific directory $HOME/.local/lib/python2.x/site-packages
# PYTHON_INSTALL_FLAGS	:= --prefix=$(HOME)/.local
## The following also specifies the user-specific directory (but not recognized by Python 2.4)
# PYTHON_INSTALL_FLAGS	:= --user
## The following will install it under $FER_DIR/lib/python2.x/site-packages
PYTHON_INSTALL_FLAGS	:= --prefix=$(FER_DIR)
## The following (empty) will install it in the system-wide package directory
## (typically /usr/lib/python2.x/site-packages)
# PYTHON_INSTALL_FLAGS	:=

## Java 1.6 jdk home directory ( $(JAVA_HOME)/bin/javac is called to build threddsBrowser.jar ).
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
