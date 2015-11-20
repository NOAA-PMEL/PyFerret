## Site-dependent definitions included in external function 
## Makefiles in an installed PyFerret directory.

## Machine type for which to build Ferret/PyFerret
##   x86_64-linux      for 64-bit RHEL
##   x86_64-linux-gnu  for 64-bit Ubuntu and many "free" Linux systems
##   i386-linux        for 32-bit RHEL
##   i386-linux-gnu    for 32-bit Ubuntu and many "free" Linux systems
## This value is used to determine which platform_specific.mk
## file to include in the Makefiles.
BUILDTYPE = $(HOSTTYPE)
# BUILDTYPE = x86_64-linux
# BUILDTYPE = x86_64-linux-gnu
# BUILDTYPE = i386-linux
# BUILDTYPE = i386-linux-gnu

## INSTALL_FER_DIR and PYTHON_EXE are only used to construct
## the location of pyferret library.  The library should be
## (for either 32-bit or 64-bit Linux)
## $(INSTALL_FER_DIR)/lib/$(PYTHON_EXE)/site-package/pyferret/libpyferret.so
## or possibly (for 64-bit Linux only)
## $(INSTALL_FER_DIR)/lib64/$(PYTHON_EXE)/site-package/pyferret/libpyferret.so

## PyFerret installation directory, usually just $(FER_DIR)
INSTALL_FER_DIR = $(FER_DIR)

## Python version used by PyFerret, either python2.6 or python2.7
PYTHON_EXE = python2.6
# PYTHON_EXE = python2.7

## FER_LOCAL_EXTFCNS is the directory in which to install
## the Ferret Fortran external functions.  The example
## functions that come with the PyFerret installation are
## installed in $(INSTALL_FER_DIR)/ext_func/pylibs
FER_LOCAL_EXTFCNS = $(INSTALL_FER_DIR)/ext_func/pylibs

##
