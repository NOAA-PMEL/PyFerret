#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# dec-00 adapted for "godae_obs" function


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

.c.o:
	cc ${GOBFLAGS} ${CFLAGS} $<

.F.o:
	$(F77) $(FFLAGS) $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(GOB_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn ocn_obs.o godae_obs.o godae_obs.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug"   "CFLAGS = $(CFLAGS) -g -Ddebug"  all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
