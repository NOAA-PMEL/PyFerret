#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
# *sh* 2/2000 - removed SAMPLEIJ from target list (replaced by SAMPLEXY)
#
# ACM 2/2001  debug macros
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
	$(F77) $(FFLAGS)  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

#all:    ferret_cmn scatter2gridgauss_xy.so scatter2gridgauss_xz.so scatter2gridgauss_yz.so scatter2gridlaplace_xy.so scatter2gridlaplace_xz.so scatter2gridlaplace_yz.so 

# all:    ferret_cmn ave_scat2grid_t.so scat2grid_t.so
all:    ferret_cmn index_to_grid.so scatgrid_nobs_xy.so scatgrid_nobs_xz.so scatgrid_nobs_xt.so scatgrid_nobs_yz.so scatgrid_nobs_yt.so scatgrid_nobs_zt.so


ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
