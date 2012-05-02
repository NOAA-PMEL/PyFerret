#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)
#
#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so

#
# Targets
#

# all:	list_good.so xcat.so ycat.so zcat.so tcat.so xreverse.so yreverse.so zreverse.so treverse.so
all:	list_good.so

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
