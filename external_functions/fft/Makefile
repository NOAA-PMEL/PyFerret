#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
# Ansley Manke: compile subroutines separately w/ fewer flags. 
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

EFN_OBJS = ef_utility/*.o

SUB_OBJS = fftsubs.o

#
# Rules
#

.f.o:
	$(F77) $(FFLAGS) -c $<

.F.so:
	f77  $(FFLAGS) -c -u  $<
	$(LD) $(LD_DYN_FLAGS) $(EFN_OBJS)  $(SUB_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn ef_utility fftsubs.o ffta.so fftp.so

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* efn ef_utility ferret_cmn

#
# End of Makefile
#
