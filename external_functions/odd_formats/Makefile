#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#   2-Jan-2001 Ansley Manke add -debug option (as in SH files)
#
# include platform specific macro definitions
#

include ../../site_specific.mk
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)
 

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.o:
	$(F77) $(FFLAGS) -c $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

all:	index_to_grid.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -C -Ddebug"   "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(MY_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
