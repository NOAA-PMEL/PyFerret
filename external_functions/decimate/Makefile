#
# Makefile for Ferret External Functions
#
#
# include platform specific macro definitions
#
include ../ef_utility/site_specific.mk
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)
 
#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

all:	piecewise.so separate.so dist3.so dist2.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
