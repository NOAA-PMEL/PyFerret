#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
# *acm* 1/2001 - add "make debug" target as per SH Makefiles
#  ACM  2/2001    use debug macros
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

.F.o:
	$(F77) $(FFLAGS) $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

all:	ferret_cmn convolvei.so convolvej.so convolvek.so convolvel.so 


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn ef_utility ef_utility

#
# End of Makefile
#
