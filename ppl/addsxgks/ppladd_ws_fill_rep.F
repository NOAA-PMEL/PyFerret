	SUBROUTINE WS_FILL_REP (zwsid,zwstype)

*	Setup the fill area indices for patterns or colors for generic WS

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.19.92 mods to support atc gks -- use/not use area bundles
*	*jd* 10.1.92 mod to support protected color indices
*	*jd* 12.3.93 mod to set color reps in metafiles as well as window ws
* 	*jd* 7.21.95 bug fix where if meta_clsplt has been invoked to close
*			the metafile, error messages are emitted when trying
*			to setup the color table for the closed ws

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
        include         'tmap_pplv11inc/gkscm1_inc.decl'
        include         'tmap_pplv11inc/gkscm2.cmn'
        include         'pplv11inc/GKSCM1.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
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

	integer		zwsid,zwstype,error
	integer		patt,ndx
	integer		max_pline,max_pmark,max_text,max_fill_area
	integer		max_pattern,max_color,usable_colors

	integer		wsmgr_reserve,line_color_reserve
	data		wsmgr_reserve/6/,line_color_reserve/6/

C	data stmt added for linux *jd* 12.96


*******************************************************************************

*	WHAT IS AVAILABLE ON THIS WORKSTATION?
	call gqlwk (zwstype,error,max_pline,max_pmark,
     .	            max_text,max_fill_area,max_pattern,max_color)
	if (error .ne. 0) goto 1000

	if (max_color .eq. 2) then
*	   USE PATTERNS -- NOT ENOUGH COLORS AVAILABLE
	   if (area_bundles) then
*	      MISSING VALUE PATTERN
	      call gsfar (zwsid,1,gpattr,2,1)

*	      LOWEST DATA PATTERN IS BLACK
	      call gsfar (zwsid,2,gsolid,1,1)

	      if (shd_levels .eq. 1) goto 30
*	      GO THRU AVAILABLE PATTERNS
	      do 10 ndx = 1,shd_levels - 1
#ifdef atc_gks
	         call gspa (1.0,1.0)
	         call gsparf (0.0,0.0)
	         patt = 1 + ndx*9/(shd_levels-1)
#else
	         patt = 28 - ndx*14/(shd_levels-1)
#endif
	         call gsfar (zwsid,ndx+2,gpattr,patt,1)
10	      continue
	   else
	      call hatch_fill_rep (zwsid)
	   endif
	else

*	   USE COLORS
*	   type *, 'protect_used_colors', protect_used_colors
	   if (protect_used_colors) then
	      lo_color_ndx(zwsid) = max(hi_color_ndx(zwsid) + 1,
     .						line_color_reserve + 1)
	      protect_used_colors = .false.
	   else
	      lo_color_ndx(zwsid) = max(line_color_reserve + 1,
     .						lo_color_ndx(zwsid))
	   endif

	   usable_colors = max_color - wsmgr_reserve - lo_color_ndx(zwsid)

	   if (usable_colors .lt. shd_levels) then
	      write (6,'(/a)') ' PPL+ error: You''re attempting to use '//
     .           'more colors than are available.'
	      write (6,'(a/)') '             Using SHASET RESET to re-use'//
     .		 ' protected colors may help.'
	      lo_color_ndx(zwsid) = max (line_color_reserve + 1,
     .				  max_color - wsmgr_reserve - shd_levels)
	   endif
	   hi_color_ndx(zwsid) = lo_color_ndx(zwsid) + shd_levels - 1

*	   type *, 'max_color, usable_colors, lo_color_ndx(zwsid),'// 
*     .						' hi_color_ndx(zwsid)'
*	   type *,  max_color, usable_colors, lo_color_ndx(zwsid), 
*     .						  hi_color_ndx(zwsid)

*	CHOOSE CUSTOM COLORS OR THE DEFAULT SELECTION
	   if (custom_colors) then
	      call setup_color_table (zwsid,lo_color_ndx(zwsid),shd_levels)
#ifdef xgks
	      if (meta_actv .and. .not. meta_clsplt) call setup_color_table 
     .			(meta_wsid,lo_color_ndx(zwsid),shd_levels)
#endif
	   else
	      call set_default_colors (zwsid,zwstype,lo_color_ndx(zwsid),
     .							shd_levels)
	   end if

	   if (.not. area_bundles) goto 30

*	   NOW SET THE FILL REP
	   do 20 ndx = 1,shd_levels
	      call gsfar (zwsid,ndx+lo_color_ndx(zwsid)+1,gsolid,1,
     .						lo_color_ndx(zwsid)-1+ndx)
20	   continue
	end if

30	return

*	FATAL ERROR
1000	return

	end
