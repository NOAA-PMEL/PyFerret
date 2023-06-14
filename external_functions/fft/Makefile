#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
# Ansley Manke: compile subroutines separately w/ fewer flags.
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# 2/2001  debug macros
# 1/2002  need $(FC) in .F.so rule
#         and compile ffta_sample, fftp_sample functions
# 7/2012  Do not need a special platform_specific file.
# 12/2022 Remove from targets list routines and functions that
#         are linked into the PyFerret program

#
# include site specific macro definitions
#
include ../ef_utility/site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)


#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = fftsubs.o


#
# Rules
#

.F.so:
	$(FC) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

all:	fftsubs.o ffta_sample.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
