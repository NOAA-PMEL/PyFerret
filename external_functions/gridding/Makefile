#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
 
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so


SUB_OBJS = gridsubs.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn ef_utility gridsubs.o scatgrid_xy.so scatgrid_xz.so scatgrid_yz.so scat2gridgauss_xy.so scat2gridgauss_xz.so scat2gridgauss_yz.so scat2gridlaplace_xy.so scat2gridlaplace_xz.so scat2gridlaplace_yz.so  


ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
