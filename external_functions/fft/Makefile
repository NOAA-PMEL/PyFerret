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

#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
include fft_platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = fftsubs.o

#
# Rules
#

.F.so:
	f77  $(FFLAGS) -c -u  $<
	$(LD) $(LD_DYN_FLAGS)   $(SUB_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn fftsubs.o ffta.so fftp.so

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* efn ferret_cmn

#
# End of Makefile
#
