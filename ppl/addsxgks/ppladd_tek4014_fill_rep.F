	SUBROUTINE TEK4014_FILL_REP (zwsid)

*	Setup the fill area indices for patterns on TEK 4014 WS

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.14.92 mod to support atc gks

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

	integer		zwsid,patt,ndx

********************************************************************************

*	MISSING VALUE IS SOLID FILL
	call gsfar (zwsid,1,gsolid,1,1)

*	GO THRU AVAILABLE HATCHING LEVELS 
	do 10 ndx = 1,shd_levels
#ifdef atc_gks
	   patt = -1000 - (ndx-1)*9/(shd_levels)
#else
	   patt = -1 - (ndx-1)*9/(shd_levels)
#endif
	   call gsfar (zwsid,ndx+1,ghatch,patt,1)
10	continue

	return

	end
