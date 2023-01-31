	SUBROUTINE WS_LINE_SPECTRUM (zwsid,zwstype, low_index)

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
*  the use of this software or in any product that INCLUDEs this
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
*	Routine to set up the line bundles for color-line plots
* PyFr *kms* 5/13 Add alpha channel (opaque)
* V691 *acm* 5/14 Fix when defining line bundles for a by-levels palette
* V694 *acm* 5/15 Fix ticket 2292 line plot thickness and ribbon plots
* 74+  *acm* 4/18 In working on DSG plotting of trajectories. Make sure enough
*                 colors are defined when using by-levels palette
* 74+  *acm* 4/18 Tweaks to color levels on ribbon plots for by-levels palettes 
*                 and centered-label key

        IMPLICIT NONE
	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'
	include 'gkscm2.cmn'
	include 'ppl_in_ferret.cmn'
        include 'ws_types.cmn'
        include 'shade_vars.cmn'
	include 'tmap_dims.parm'
        include 'xplot_setup.cmn'
        INCLUDE 'gkspar.inc'
        INCLUDE 'parampl5_dat.decl'
        INCLUDE 'PARAMPL5.DAT'
        INCLUDE 'plt_inc.decl'
        INCLUDE 'PLT.INC'
        INCLUDE 'pen_inc.decl'  ! with num_line_colors
        INCLUDE 'PEN.INC'
	INCLUDE 'save_colors.cmn'
	INCLUDE 'errmsg.parm'

	integer		zwsid,zwstype,i,j, low_index
	integer 	ltype(6)
	data		ltype/1,2,3,4,-3,-4/

	integer		max_pline, max_pmark, max_text, max_fill_area
	integer		max_pattern, max_color, error, ncolors
	integer		fktr, index, line_color_reserve,
     .			icolr, ndx, n1, n2, n3, slen,
     .			nthick, usable_colors, ncolors_to_set, status
	real		thickness, frac, r_val, g_val, b_val, a_val

	CHARACTER*100   err_msg
	CHARACTER*48	TM_FMT, note

	integer		max_bundles
	parameter	(max_bundles = 250)  ! matches MAX_BUNDL_TBL in xgks

*******************************************************************************
*	WHAT IS AVAILABLE ON THIS WORKSTATION?
	call FGD_GQLWK (zwstype,error,max_pline,max_pmark,
     .	            max_text,max_fill_area,max_pattern,max_color)
	if (error .ne. 0) goto 1000

*	COLOR OR MONOCHROME?
	if (max_color .le. 2) goto 100
	
* ncolors_to_set
	 ncolors_to_set = shd_levels
	 if (rgb_mapping .eq. by_level) ncolors_to_set = ncl

*	COLOR WORKSTATIONS:

* Set up a color spectrum in the remaining space.

	IF ((max_color - num_line_colors) .LT. 0) GOTO 200

	line_color_reserve = num_line_colors
	
*   USE COLORS
        if (protect_used_colors) then
           lo_color_ndx(zwsid) = MAX(hi_color_ndx(zwsid) + 1, line_color_reserve + 1)
           protect_used_colors = .false.
        else
           lo_color_ndx(zwsid) = MAX(line_color_reserve + 1, lo_color_ndx(zwsid))
        endif

        usable_colors = max_color - line_color_reserve - lo_color_ndx(zwsid)

        if (usable_colors .lt. ncolors_to_set+1) then
           write (6,'(/a)') ' PPL warning: You''re attempting to use '//
     .           'more colors than are available.'
           write (6,'(a/)') '             Using PPL SHASET RESET to re-use'//
     .		 ' protected colors may help.'
           lo_color_ndx(zwsid) = max (line_color_reserve + 1,
     .       max_color - line_color_reserve - (ncolors_to_set+1))
        endif
        hi_color_ndx(zwsid) = lo_color_ndx(zwsid) + (ncolors_to_set+1) - 1

* Plus one more for the bad-value color (e.g. missing data drawn as gray)

        hi_color_ndx(zwsid) = hi_color_ndx(zwsid) + 1
	bad_ribbon_ndx = (ncolors_to_set+1) + 1

*	CHOOSE CUSTOM COLORS OR THE DEFAULT SELECTION
        if (custom_colors) then
           call setup_color_table (zwsid,lo_color_ndx(zwsid),ncolors_to_set)
        else
           call set_default_colors (zwsid,zwstype,lo_color_ndx(zwsid),
     .							ncolors_to_set)
        end if


* Set line bundles
* The low-index preserves line and thickness defintions for the standard line-plot colors.

	nthick = 3
	low_index = MAX(num_line_colors* nthick + 2, lo_color_ndx(zwsid)) 

* GKS has a hard-wired MAX_BUNDL_TBL set at 250. This means we are
* limited on number of colors drawn with thick lines.

        ncolors = ncolors_to_set
	IF ((low_index+(nthick-1)*(ncolors+1)+ncolors) .GT. max_bundles) THEN

	    n3 = (low_index+(nthick-1)*(ncolors+1)+ncolors)
	    n2 = (low_index+(nthick-2)*(ncolors+1)+ncolors)
	    n1 = (low_index+(nthick-3)*(ncolors+1)+ncolors)

	    note = TM_FMT(DBLE(max_bundles), 0, 12, slen)

	    IF (n1. GT. max_bundles) THEN
	       CALL WARN ('Graphics limits settings to '//note(:slen)//
     .	       ' line/thickness combinations plus existing spectrum colors')
	       note = TM_FMT(DBLE(ncolors), 0, 12, slen)
	       CALL WARN ('Cannot draw /THICK=1 lines with current number '//
     .	       'of color levels = '//note(:slen) )
     
	    ELSE IF (n2. GT. max_bundles) THEN
	       CALL WARN ('Graphics limits settings to '//note(:slen)//
     .	       ' line/thickness combinations plus existing spectrum colors')
	       note = TM_FMT(DBLE(ncolors), 0, 12, slen)
	       CALL WARN ('Cannot draw /THICK=2 lines with current number '//
     .	       'of color levels = '//note(:slen)//'; THICK=1 will be ok' )

	    ELSE IF (n3. GT. max_bundles) THEN
	       CALL WARN ('Graphics limits settings to '//note(:slen)//
     .	       ' line/thickness combinations plus existing spectrum colors')
	       note = TM_FMT(DBLE(ncolors), 0, 12, slen)
	       CALL WARN ('Cannot draw /THICK=3 lines with current number '//
     .	       'of color levels = '//note(:slen) //'; THICK=1 or 2 will be ok' )
	    ENDIF

        ENDIF

        if (rgb_mapping .eq. by_level) ncolors = ncl
	do 20 j = 1, ncolors
	   do 10 fktr = 1,nthick     ! thickness
              thickness = REAL(fktr)
              icolr = j + lo_color_ndx(zwsid) - 1
              index = low_index + (fktr-1) * (ncolors+1) + j
              CALL FGD_GSPLR(zwsid, index, 1, thickness, icolr)
10	   continue
20	continue

* Initialize the color for bad-data asc

        r_val = 0.75
        g_val = 0.75
        b_val = 0.75
        a_val = 1.00

* What is ribbon_miss?  If user asked for a palette with /missing= 
* then set that color.

	IF (ribbon_miss(1) .GE. 0.) THEN
           r_val = ribbon_miss(1)/100.
           g_val = ribbon_miss(2)/100.
           b_val = ribbon_miss(3)/100.
           a_val = ribbon_miss(4)/100.
	ENDIF
	ndx = lo_color_ndx(zwsid)+ncolors
	CALL FGD_GSCR(zwsid,ndx,r_val,g_val,b_val,a_val)

	do 40 j = ncolors + 1, ncolors + 1
	   do 30 fktr = 1,nthick     ! thickness
              thickness = REAL(fktr)
              icolr = j + lo_color_ndx(zwsid) - 1
              index = low_index + (fktr-1) * (ncolors_to_set+1) + j
              CALL FGD_GSPLR(zwsid, index, 1, thickness, icolr)
30	   continue
40	continue

* ERROR
1000	return

 100    err_msg = 'Not a color worksatation, cannot set spectrum for color lines'
        CALL errmsg (ferr_out_of_range, status, err_msg, *1000)

 200    err_msg = 'Too many colors in color spectrum for ribbon plot'
        CALL errmsg (ferr_out_of_range, status, err_msg, *1000)
	end
