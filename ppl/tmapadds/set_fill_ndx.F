	SUBROUTINE SET_FILL_NDX (fill_ndx)

*	Sets fill index appropriately.  DEC uses bundled indices,
*	at this date ATC has too few to use them, so must use individual
*	attributes.

*	J Davison 10.12.92
*       *jd* 7.31.97 mode to change call to gqlwk to query_xgks_wkstn
*       *jd* 10.98 handle "atomic" colors
* 3/99 *kob* 	move var init off of decl list  - cleaning up after jerry.

#ifdef unix
        include         'shade_vars.cmn'
	include		'gkscm1_inc.decl'
	include		'GKSCM1.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	INCLUDE		'tmap_pplv11inc:tmap_gkscm1.inc'
#endif

#   ifdef unix
        include        'gkspar.inc'
#   else
        INCLUDE        'sys$library:gksdefs.bnd'
#   endif

	integer		fill_ndx,style,pattern_ndx,solid
	DATA            solid/21/
        integer         max_pline,max_pmark,max_text,max_fill_area
        integer         max_pattern,max_color,error,color_ndx
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
*	      CAN STILL USE COLOR INDICES -- BUT INDIVIDUAL
	      
*             USE ATOMIC COLORS?
	      if (rgb_mapping .ne. by_level) then
*                SMOOTH OUT COLORS TO FULL RANGE
	         call gsfaci (fill_ndx+lo_color_ndx(wsid)-2)
	      else
*                USE ATOMIC COLORS -- REUSE AS NECESSARY
		 color_ndx = mod((fill_ndx - 2),ncl) + 2
		 call gsfaci (color_ndx+lo_color_ndx(wsid)-2)
	      end if
	   else
#ifdef atc_gks
*	      OTHERWISE USE HATCHING SCHEME
	      incr = real (fill_ndx/2) - 1.0

*	      SEQUENTIAL INDICES ARE HATCHING ROTATED 90 DEGREES
	      if (mod(fill_ndx,2) .eq. 0) then
	            style = -1005
		    spc = min_spc + incr*delta_spc
	      else
		    style = -1004
		    spc = (min_spc + incr*delta_spc) * 0.707
	      endif

	      call gsfasi (style)
	      call guesc061 (wsid,1,spc)
#endif
	   endif
	endif

#ifdef xgks
	pattern_ndx = mod((fill_ndx - 2),npat) + 1

	if (patterns(pattern_ndx) .eq. solid) then
	   call gsfais (GSOLID)
	else
	   call gsfais (GHATCH)
	   call gsfasi (-1 * patterns(pattern_ndx))
	end if
#endif

	return
	end

