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
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
include fft_platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = fftinv_subs.o

#
# Rules
#

.F.so:
	$(F77)  $(FFLAGS)  $<
	$(LD) $(LD_DYN_FLAGS)   $(SUB_OBJS) $*.o -o $*.so 

#
# Targets
#

# all:	ferret_cmn fftinv_subs.o fft_im.so fft_re.so fft_inverse.so lsl_lowpass.so fft_amp.so fft_phas.so

all:	ferret_cmn 

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* efn ferret_cmn

#
# End of Makefile
#
