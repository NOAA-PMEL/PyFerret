#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
 
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

EFN_OBJS = ef_utility/*.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(EFN_OBJS) $*.o -o $*.so 
	cp $@ $(FER_EXTERNAL_FUNCTIONS)

#
# Targets
#

all:	ef_utility ferret_cmn  ffta.so fftp.so

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ef_utility/ferret_cmn ferret_cmn

clean:
	-rm -f *.o *.so core a.out temp.* efn ferret_cmn

#
# End of Makefile
#
