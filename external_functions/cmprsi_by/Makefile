#
# Makefile for Ferret External Functions
#
# *sh* 12/27/00
#
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.o:
	$(F77) $(FFLAGS) -u -c $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

all:	ferret_cmn cmprsi_by.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -C -Ddebug"   "CFLAGS = $(CFLAGS) -g -Ddebug"  all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
