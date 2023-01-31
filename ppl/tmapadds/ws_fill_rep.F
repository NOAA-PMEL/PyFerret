	SUBROUTINE WS_FILL_REP (zwsid,zwstype)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
*	Setup the fill area indices for patterns or colors for generic WS

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.19.92 mods to support atc gks -- use/not use area bundles
*	*jd* 10.1.92 mod to support protected color indices
*	*jd* 12.3.93 mod to set color reps in metafiles as well as window ws
* 	*jd* 7.21.95 bug fix where if meta_clsplt has been invoked to close
*			the metafile, error messages are emitted when trying
*			to setup the color table for the closed ws
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn
* V600 *acm* 5/06  num_line_colors for MODE LINECOLORS. Number of line colors 
*                  no longer fixed at 6.

        IMPLICIT NONE
        include         'shade_vars.cmn'
        include         'gkscm1_inc.decl'
        include         'gkscm2.cmn'
        include         'GKSCM1.INC'
	include         'gkspar.inc'
        include         'parampl5_dat.decl'
        include         'PARAMPL5.DAT'
        include         'plt_inc.decl'
        include         'PLT.INC'
	include		'pen_inc.decl'
        include         'PEN.INC'  ! with num_linecolors

	integer		zwsid,zwstype,error
	integer		patt,ndx
	integer		max_pline,max_pmark,max_text,max_fill_area
	integer		max_pattern,max_color,usable_colors

	integer		wsmgr_reserve,line_color_reserve
C	data		wsmgr_reserve/6/,line_color_reserve/6/

C	data stmt added for linux *jd* 12.96


*******************************************************************************

        line_color_reserve = num_line_colors
        wsmgr_reserve = num_line_colors

*	WHAT IS AVAILABLE ON THIS WORKSTATION?
	call FGD_GQLWK (zwstype,error,max_pline,max_pmark,
     .	            max_text,max_fill_area,max_pattern,max_color)
	if (error .ne. 0) goto 1000

	if (max_color .eq. 2) then
*	   USE PATTERNS -- NOT ENOUGH COLORS AVAILABLE
	   if (area_bundles) then
*	      MISSING VALUE PATTERN
	      call FGD_GSFAR(zwsid,1,gpattr,2,1)

*	      LOWEST DATA PATTERN IS BLACK
	      call FGD_GSFAR(zwsid,2,gsolid,1,1)

	      if (shd_levels .eq. 1) goto 30
*	      GO THRU AVAILABLE PATTERNS
	      do 10 ndx = 1,shd_levels - 1
	         patt = 28 - ndx*14/(shd_levels-1)
	         call FGD_GSFAR(zwsid,ndx+2,gpattr,patt,1)
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
c	      write (6,'(/a)') ' PPL+ error: You''re attempting to use '//
	      write (6,'(/a)') ' PPL warning: You''re attempting to use '//
     .           'more colors than are available.'
	      write (6,'(a/)') '             Using PPL SHASET RESET to re-use'//
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
	   else
	      call set_default_colors (zwsid,zwstype,lo_color_ndx(zwsid),
     .							shd_levels)
	   end if

	   if (.not. area_bundles) goto 30

*	   NOW SET THE FILL REP
	   do 20 ndx = 1,shd_levels
	      call FGD_GSFAR(zwsid, ndx+lo_color_ndx(zwsid)+1, gsolid,
     .                       1, lo_color_ndx(zwsid)-1+ndx)
20	   continue
	end if

30	return

*	FATAL ERROR
1000	return

	end
