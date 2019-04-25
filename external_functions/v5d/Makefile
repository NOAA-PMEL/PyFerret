#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#  ACM 2/2001  debug macros

#
# include site specific macro definitions
#
include ../ef_utility/site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)
include v5d_platform_specific_flags.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so

V5D_OBJS = binio.o v5d.o

#
# Rules
#

.c.o:
	$(CC) $(V5DFLAGS) $(CFLAGS) -c $<

.F.so:
	$(FC) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(V5D_OBJS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

all:	binio.o v5d.o writev5d.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
