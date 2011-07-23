## Site-dependent definitions included in Makefiles

## Machine for which to build Ferret
## Use $(HOSTTYPE) to build natively for the machine you are using
BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= i386-linux

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
# READLINE_DIR	= /
# READLINE_DIR	= /usr
READLINE_DIR	= /usr/local

## Installation directory for libz static library
## (contains include and lib or lib64 subdirectories)
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
