	SUBROUTINE HATCH_FILL_REP (zwsid)

*	Setup the fill area indices for hatching

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.14.92 mod to support atc gks
*	*jd* 10.12.92 mod for new hatching scheme


#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
        include        'tmap_pplv11inc/shade_vars.cmn'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
        INCLUDE        'tmap_pplv11inc:tmap_shade_vars.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
        include         'tmap_pplv11inc/shade_vars.cmn'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
        INCLUDE        'tmap_pplv11inc:tmap_shade_vars.inc'
#   endif
#endif

	integer		zwsid,patt,ndx

*******************************************************************************


*	USE AREA BUNDLES IF SET
	if (area_bundles) then

*	   MISSING VALUE IS SOLID FILL
	   call gsfar (zwsid,1,gsolid,1,1)

*	   GO THRU AVAILABLE HATCHING LEVELS 
	   do 10 ndx = 1,shd_levels
#ifdef atc_gks
	      patt = -1000 - (ndx-1)*9/(shd_levels)
#else
	      patt = -1 - (ndx-1)*9/(shd_levels)
#endif
	      call gsfar (zwsid,ndx+1,ghatch,patt,1)
10	   continue
	else
#ifdef atc_gks
	   call guesc020 (zwsid,gsalws)
#endif
	   call gsfais (ghatch)
	   call gsfaci (1)
	endif

	return

	end
