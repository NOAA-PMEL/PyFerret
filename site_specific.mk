## Site-dependent definitions included in Makefiles

## Machine for which to build Ferret
## Use $(HOSTTYPE) to build natively for the machine you are using
BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= i386-linux
# BUILDTYPE	= i386-apple-darwin

## Installation directory for built Ferret.  Using the "install"
## Makefile target circumvents the need to create the fer_*.tar.gz
## files just for creating a Ferret installation.
#INSTALL_FER_DIR = $(FER_DIR)
INSTALL_FER_DIR = /home/users/ansley/ferret_distributions/rhel6_64

## Installation directory for HDF5 static libraries
## (contains include and lib or lib64 subdirectories)
# HDF5_DIR	= /usr
# HDF5_DIR	= /usr/local
HDF5_DIR	= /usr/local/hdf5_189_64

## Installation directory for NetCDF static libraries
## (contains include and lib or lib64 subdirectories)
# NETCDF4_DIR	= /usr
# NETCDF4_DIR	= /usr/local
NETCDF4_DIR	= /usr/local/netcdf_432_64
NETCDF4_DIR	= /home/users/ansley/local/flat_64_nc4331

## Installation directory for readline static libraries
## (contains include and lib or lib64 subdirectories)
# READLINE_DIR	= /
READLINE_DIR	= /usr
#READLINE_DIR	= /usr/local
#READLINE_DIR	= /home/users/tmap/flat_64/readline-4.1

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
