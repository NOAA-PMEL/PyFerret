#
# Makefile for Ferret External Functions
#
# 2/01 *sh*

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

DAT_OBJS = flag2ddups.o

#
# Rules
#

.c.so:
	cc ${DATFLAGS} ${CFLAGS} $<
	$(LD) $(LD_DYN_FLAGS) $(DAT_OBJS) $*.o -o $*.so 

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $(DAT_OBJS) $*.o -o $*.so 

#
# Targets
#

# all:	$(DAT_OBJS) scat2ddups.o scat2ddups.so
all:	

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
