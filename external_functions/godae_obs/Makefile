#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
# dec-00 adapted for "godae_obs" function
# ACM 2/2001

#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

GOB_OBJS = ocn_obs.o

#
# Rules
#

.c.so:
	cc ${GOBFLAGS} ${CFLAGS} $<
	$(LD) $(LD_DYN_FLAGS) $(GOB_OBJS) $*.o -o $*.so 

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $(GOB_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn ocn_obs.o godae_obs.o godae_obs.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
