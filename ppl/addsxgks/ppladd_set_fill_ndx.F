	SUBROUTINE SET_FILL_NDX (fill_ndx)

*	Sets fill index appropriately.  DEC uses bundled indices,
*	at this date ATC has too few to use them, so must use individual
*	attributes.

*	J Davison 10.12.92
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
	include		'tmap_pplv11inc/gkscm1_inc.decl'
	include		'pplv11inc/GKSCM1.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	INCLUDE		'tmap_pplv11inc:tmap_gkscm1.inc'
#endif

	integer		fill_ndx,style
        integer         max_pline,max_pmark,max_text,max_fill_area
        integer         max_pattern,max_color,error
	real		incr,spc,min_spc,delta_spc
  
	parameter	(min_spc=0.05,delta_spc=0.01)

*******************************************************************************

	if (area_bundles) then
*	   USE SET FILL AREA INDEX
	   call gsfai (fill_ndx+lo_color_ndx(wsid))
	else
*	   HOW MANY COLORS HAVE WE?
           call query_xgks_wkstn (wstype,error,max_pline,max_pmark,
     .              max_text,max_fill_area,max_pattern,max_color)

	if (max_color .gt. 2) then 

*	   CAN STILL USE COLOR INDICES -- BUT INDIVIDUAL
	      call gsfaci (fill_ndx+lo_color_ndx(wsid)-2)
	   else

*	   OTHERWISE USE HATCHING SCHEME
	   incr = real (fill_ndx/2) - 1.0

*	   SEQUENTIAL INDICES ARE HATCHING ROTATED 90 DEGREES
	   if (mod(fill_ndx,2) .eq. 0) then
	         style = -1005
		 spc = min_spc + incr*delta_spc
	      else
		 style = -1004
		 spc = (min_spc + incr*delta_spc) * 0.707
	      endif

	      call gsfasi (style)
#ifdef atc_gks
	      call guesc061 (wsid,1,spc)
#endif
	   endif

	endif

	return
	end

