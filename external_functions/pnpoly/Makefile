#
# Makefile for Ferret External Functions
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

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

#all:	ferret_cmn pnpoly_subs.o pt_in_poly.so pt_in_poly_curvi.so
all:	ferret_cmn pnpoly_subs.o pt_in_poly.so


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
