	SUBROUTINE TEK4107_LINE_BUNDLE (zwsid)

*	Routine to set up the line bundles for tek4107's
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* mod to support atc gks                                                                         
#ifdef unix
	include		'tmap_pplv11inc/gkscm2.cmn'
#else
	INCLUDE		'tmap_pplv11inc:tmap_gkscm2.inc'
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

	integer		zwsid,ndx,fktr
	integer		index(6)
	data		index /1,3,6,9,11,13/

C	data statement above added 12.96 *jd* for linux port

*******************************************************************************

*	TEK4107 HAS LIMITED COLOR INDICES -- USE AVAILABLE COLORS
	do 20 ndx = 1,6
	   do 10 fktr = 1,3
	      call gsplr (zwsid,(fktr-1)*6+ndx,1,real(fktr),index)
10	   continue
20	continue

	return
	end

