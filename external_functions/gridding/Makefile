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
#SUB_OBJS = nobs.o

#
# Rules
#


.F.so:
	$(F77) $(FFLAGS) -g -c  $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so
 
.c.o:
	cc ${CFLAGS} -c $<

.f.o:
	$(F77) $(FFLAGS) $<

#
# Targets
#
#all:    scatter2gridgauss_xy.so scatter2gridgauss_xz.so scatter2gridgauss_yz.so scatter2gridlaplace_xy.so scatter2gridlaplace_xz.so scatter2gridlaplace_yz.so 

# all:    ave_scat2grid_t.so scat2grid_t.so
# all:    nobs.o index_to_grid.so scatgrid_nobs_xy.so scatgrid_nobs_xz.so scatgrid_nobs_xt.so scatgrid_nobs_yz.so scatgrid_nobs_yt.so scatgrid_nobs_zt.so scat2grid_bin_xy.so scat2grid_nobs_xt.so scat2grid_nobs_xz.so scat2grid_nobs_yt.so scat2grid_nobs_yz.so scat2grid_nobs_zt.so

all:	tracks2grid_var_xyt.so tracks2grid_ave_xyt.so 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so *.f

#
# End of Makefile
#
