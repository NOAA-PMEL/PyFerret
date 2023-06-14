#
# Makefile for Ferret External Functions
#
# 12/2022 Remove from targets list routines and functions that
#         are linked into the PyFerret program
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
	$(FC) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

# all:	fc_isubset.so expnd_by_len.so
all:	
 
debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
