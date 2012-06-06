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
# 1/2002  need $(F77) in .F.so rule
#         and compile ffta_sample, fftp_sample functions

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

include fft_platform_specific_flags.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = fftinv_subs.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

all:	fftinv_subs.o fftsubs.o fft_amp.so fft_phas.so ffta_sample.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
