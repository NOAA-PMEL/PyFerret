#ifdef FORTRAN_90

	BLOCK DATA YABD

*	Yet another block data, for linux/f90 install
*	Initialize ppl_in_ferret here
*	*jd* 12.96

* special COMMON to let this routine know if FERRET is in control
#ifdef unix
	include 'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
	INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

	DATA	PPL_in_FERRET / .FALSE. /	

	end

#endif

