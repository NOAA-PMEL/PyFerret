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

 #
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c -u $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn ef_utility convolvei.so convolvej.so convolvek.so convolvel.so 

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn ef_utility ef_utility

#
# End of Makefile
#
