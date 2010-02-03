#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# ACM 2/2001  debug macros
#
#  include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 
#
# Macros
#

.SUFFIXES: .so
#SUB_OBJS = nobs.o

#
# Rules
#


.F.so:
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so
 
.c.o:
	cc ${CFLAGS} -c $<

.f.o:
	$(F77) $(FFLAGS) $<

#
# Targets
#
#all:    ferret_cmn scatter2gridgauss_xy.so scatter2gridgauss_xz.so scatter2gridgauss_yz.so scatter2gridlaplace_xy.so scatter2gridlaplace_xz.so scatter2gridlaplace_yz.so 

# all:    ferret_cmn ave_scat2grid_t.so scat2grid_t.so
# all:    ferret_cmn nobs.o index_to_grid.so scatgrid_nobs_xy.so scatgrid_nobs_xz.so scatgrid_nobs_xt.so scatgrid_nobs_yz.so scatgrid_nobs_yt.so scatgrid_nobs_zt.so scat2grid_bin_xy.so

all:	ferret_cmn tracks2grid_ave_xyt.so scatgrid_bin_xyt.so scatgrid_nobs_xyt.so  scatgrid_nbin_xyt.so 
more:    ferret_cmn scatgrid_bin_xyt.so scat2gridgauss_xy_fix.so scat2grid_bin_xy.so scatgrid_nobs_xyt.so scatgrid_nbin_xy.so scatgrid_nbin_xyt.so 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so *.f core a.out temp.* ferret_cmn

#
# End of Makefile
#
