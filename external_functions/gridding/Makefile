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

include ../ef_utility/site_specific.mk
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so
#SUB_OBJS = nobs.o
SUB_OBJS = modscatminmax.o nobsxytv.o scat2grid_subs_mask.o
#SUB_OBJS = scat2grid_subs.o

#
# Rules
#

.F.so:
	$(FC) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $(SYSLIBS) $*.o -o $*.so
 
.c.o:
	$(CC) $(CFLAGS) -c $<

.F.o:
	$(FC) $(FFLAGS) -c  $<

#
# Targets
#
#all:    scatter2gridgauss_xy.so scatter2gridgauss_xz.so scatter2gridgauss_yz.so scatter2gridlaplace_xy.so scatter2gridlaplace_xz.so scatter2gridlaplace_yz.so 

# all:    ave_scat2grid_t.so scat2grid_t.so
# all:    nobs.o index_to_grid.so scatgrid_nobs_xy.so scatgrid_nobs_xz.so scatgrid_nobs_xt.so scatgrid_nobs_yz.so scatgrid_nobs_yt.so scatgrid_nobs_zt.so scat2grid_bin_xy.so scat2grid_nobs_xt.so scat2grid_nobs_xz.so scat2grid_nobs_yt.so scat2grid_nobs_yz.so scat2grid_nobs_zt.so

all:	$(SUB_OBJS) tracks2grid_std_xyt.so scat2grid_std_xyt.so \
	scat2grid_minmax_xyt.so scat2grid_mask_bin_xyt.so scat2grid_mask_minmax_xyt.so \
	scat2grid_mask_nbin_xyt.so scat2grid_mask_std_xyt.so tracks2grid_mask_ave_xyt.so

#all:	$(SUB_OBJS) index_to_grid.so scat2grid_nobs_xy.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
