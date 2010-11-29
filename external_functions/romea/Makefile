#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#  ACM  2/2001  debug macros
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = rr_density.o 

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS)  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $(SUB_OBJS) $*.o -o $*.so
 

#
# Targets
#


all:	ferret_cmn rr_density.o ffta_2.so ffta_3.so ffta_sample.so fftabst.so rr_adv_sx.so rr_adv_sy.so rr_adv_sz.so rr_adv_tx.so rr_adv_ty.so rr_adv_tz.so rr_adv_umetric.so rr_adv_ux.so rr_adv_uy.so rr_adv_uz.so rr_adv_vmetric.so rr_adv_vx.so rr_adv_vy.so rr_adv_vz.so rr_av.so rr_cross_spectrum.so rr_cross_spectrum_abs.so rr_diff_sx.so rr_diff_sy.so rr_diff_sz.so rr_diff_tx.so rr_diff_ty.so rr_diff_tz.so rr_diff_umetric.so rr_diff_ux.so rr_diff_uy.so rr_diff_uz.so rr_diff_vmetric.so rr_diff_vx.so rr_diff_vy.so rr_diff_vz.so rr_dsdt.so rr_dtdt.so rr_dudt.so rr_dvdt.so rr_flux_sx.so rr_flux_sy.so rr_flux_sz.so rr_flux_tx.so rr_flux_ty.so rr_flux_tz.so rr_flux_ux.so rr_flux_uy.so rr_flux_uz.so rr_flux_vx.so rr_flux_vy.so rr_flux_vz.so rr_fv.so rr_geopotential_height.so rr_grad_pbc_u.so rr_grad_pbc_v.so rr_grad_ps_u.so rr_grad_ps_v.so rr_taux.so rr_tauy.so rr_uux_qiao.so rr_vuy_qiao.so rr_w_qiao.so rr_wuz_qiao.so 

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
