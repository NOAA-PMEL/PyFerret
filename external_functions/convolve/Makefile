#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
# *acm* 1/2001 - add "make debug" target as per SH Makefiles
#  ACM  2/2001    use debug macros

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../../platform_specific.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so

#
# Targets
#

# all:	convolvei.so convolvej.so convolvek.so convolvel.so 
all:	


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
