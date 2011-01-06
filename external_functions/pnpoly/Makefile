#
# Makefile for Ferret External Functions
#

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


SUB_OBJS = pnpoly_subs.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -g $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS)  $*.o -o $*.so 

.c.o:
	cc ${CFLAGS} $<

.f.o:
	$(F77) $(FFLAGS) $<

#
# Targets
#

#all:	pnpoly_subs.o pt_in_poly.so pt_in_poly_curvi.so
all:	pnpoly_subs.o pt_in_poly.so


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
