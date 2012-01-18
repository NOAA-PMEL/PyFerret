#
# Makefile for Ferret External Functions
#
# include platform specific macro definitions
#
include ../../site_specific.mk
include ../ef_utility/platform_specific_flags.mk.$(BUILDTYPE)
 
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

all:	ferret_cmn fc_isubset.so expnd_by_len.so
 
debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all


ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
