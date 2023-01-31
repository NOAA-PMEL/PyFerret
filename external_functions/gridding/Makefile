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
# 12/2022 Remove from targets list routines and functions that
#         are linked into the PyFerret program

#
# include site specific macro definitions
#
include ../ef_utility/site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so
SUB_OBJS = modscatminmax.o nobsxytv.o scat2grid_subs_mask.o  scat2grid_subs.o
#SUB_OBJS = scat2grid_subs.o

#
# Rules
#

.F.so:
	$(FC) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o $(SYSLIBS) -o $*.so

.F.o:
	$(FC) $(FFLAGS) -c  $<

#
# Targets
#

#all:	$(SUB_OBJS) index_to_grid.so scat2grid_nobs_xy.so
#all:	$(SUB_OBJS) tracks2grid_std_xyt.so scat2grid_std_xyt.so \
#	scat2grid_minmax_xyt.so scat2grid_mask_bin_xyt.so scat2grid_mask_minmax_xyt.so \
#	scat2grid_mask_nbin_xyt.so scat2grid_mask_std_xyt.so tracks2grid_mask_ave_xyt.so

all:	$(SUB_OBJS)  \
	scat2grid_mask_bin_xyt.so scat2grid_mask_minmax_xyt.so \
	scat2grid_mask_nbin_xyt.so scat2grid_mask_std_xyt.so 


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
