## Site-dependent definitions included in Makefiles

## Temporary hacks to use old readline libraries and
## netcdf4.1.2 and hdf1.8.6 libraries
## on RH6 machines these are installed in local ansley directories

## Machine for which to build Ferret
## Use $(HOSTTYPE) to build natively for the machine you are using
BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= i386-linux

## Installation directory for built Ferret.  Using the "install"
## Makefile target circumvents the need to create the fer_*.tar.gz
## files just for creating a Ferret installation.
INSTALL_FER_DIR = $(FER_DIR)

## Installation directory for HDF5 static libraries
## (contains include and lib or lib64 subdirectories)
# HDF5_DIR	= /usr
# HDF5_DIR	= /usr/local
# HDF5_DIR	= /home/users/tmap/flat_32
# HDF5_DIR	= /home/users/tmap/flat_64
# HDF5_DIR	= /usr/local/hdf5_186
HDF5_DIR	= /usr/local/hdf5_187

## Installation directory for NetCDF static libraries
## (contains include and lib or lib64 subdirectories)
# NETCDF4_DIR	= /usr
# NETCDF4_DIR	= /usr/local
# NETCDF4_DIR	= /home/users/tmap/flat_32
# NETCDF4_DIR	= /home/users/tmap/flat_64
NETCDF4_DIR	= /usr/local/netcdf_412
# NETCDF4_DIR	= /usr/local/netcdf_413
# NETCDF4_DIR	= /home/users/ansley/local/linux_nc42

## Installation directory for readline static libraries
## (contains include and lib or lib64 subdirectories)
## Version 6.x needed for PyFerret
# READLINE_DIR	= /
# READLINE_DIR	= /usr/local
# READLINE_DIR	= /home/users/tmap/flat_32/readline-4.1
# READLINE_DIR	= /home/users/tmap/flat_64/readline-4.1
READLINE_DIR	= /usr

## Installation directory for libz static library
## (contains include and lib or lib64 subdirectories)
## Version 1.2.5 recommended by NetCDF
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
