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

#
# Targets
#

all:	ef_utility ferret_cmn zaxreplace_avg.so

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
