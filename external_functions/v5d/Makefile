#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
 
#
# include platform specific macro definitions
#

include ../examples/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

EFN_OBJS = efn/*.o

V5D_OBJS = binio.o v5d.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(EFN_OBJS) $(V5D_OBJS) $*.o -o $*.so 
	cp $@ $(FER_EXTERNAL_FUNCTIONS)

#
# Targets
#

all:	efn ferret_cmn binio.o v5d.o writev5d.so

efn:
	ln -s ../examples/efn efn

ferret_cmn:
	ln -s ../examples/efn/ferret_cmn ferret_cmn

clean:
	-rm -f *.o *.so core a.out temp.* efn ferret_cmn

#
# End of Makefile
#
