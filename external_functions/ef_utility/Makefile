#
# Site-specific defines
#
include site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include platform_specific.mk.$(BUILDTYPE)

#
# The COPY_FERRET_EF_MEM_SUBSC subroutine needs to be called at the start 
# of the compute method of any PyFerret external function written in Fortran 
# in order to make an external copy of the (Py)Ferret internal common block 
# FERRET_EF_MEM_SUBSC.  Because the PyFerret module is shared-object library 
# loaded privately by Python, the Ferret common blocks are not visible to 
# these external functions. 
#

#
# Targets
#

.PHONY: all
all: copy_ferret_ef_mem_subsc.o

.PHONY: install
install:

.PHONY: clean
clean:
	rm -f *.o

#
# End of Makefile
#
