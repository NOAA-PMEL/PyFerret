#
# Makefile for Ferret External Functions
#
# lanczos function from Brett McDaniel was in
# Fortran 90.  Translated to f77 by Ansley Manke  6/25/04
#
#  remove refs to ef_utility/*.o  (now in ferret executable)

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so

#
# Targets
#

# all:    lanczos.so
all:


install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
