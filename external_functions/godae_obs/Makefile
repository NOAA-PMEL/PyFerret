#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
# dec-00 adapted for "godae_obs" function
# ACM 2/2001

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

all:	ocn_obs.o godae_obs.o godae_obs.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
