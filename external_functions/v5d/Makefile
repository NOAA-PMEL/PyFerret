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
include v5d_platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

EF_OBJS = ef_utility/*.o

V5D_OBJS = binio.o v5d.o

#
# Rules
#

.c.o:
	cc ${V5DFLAGS} ${CFLAGS} -c $<

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(EF_OBJS) $(V5D_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ef_utility ferret_cmn binio.o v5d.o writev5d.so

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_EXTERNAL_FUNCTIONS)

clean:
	-rm -f *.o *.so core a.out temp.* ef_utility ferret_cmn

#
# End of Makefile
#
