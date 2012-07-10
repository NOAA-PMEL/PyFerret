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
include ../ef_utility/site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so
#SUB_OBJS = modscatminmax.o nobsxytv.o scat2grid_subs_mask.o
SUB_OBJS = scat2grid_subs.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

# all:	$(SUB_OBJS) tracks2grid_std_xyt.so tracks2grid_ave_xyt.so scat2grid_std_xyt.so \
#	scat2grid_minmax_xyt.so scat2grid_mask_bin_xyt.so scat2grid_mask_minmax_xyt.so \
#	scat2grid_mask_nbin_xyt.so scat2grid_mask_std_xyt.so tracks2grid_mask_ave_xyt.so \
#	tracks2grid_mask_std_xyt.so tracks2grid_mask_var_xyt.so

all:	$(SUB_OBJS) scat2grid_nobs_xy.o

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
