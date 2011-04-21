#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# ACM 2/2001  debug macros

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../../platform_specific.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so

#
# Targets
#

all:	tracks2grid_std_xyt.so tracks2grid_ave_xyt.so scat2grid_std_xyt.so scat2grid_minmax_xyt.so

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so *.f

#
# End of Makefile
#
