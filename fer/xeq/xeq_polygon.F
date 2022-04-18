	SUBROUTINE XEQ_POLYGON

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
* execute the POLYGON command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500:  *sh* and *jd*
*        *jd  fix for v5.01 for fill if /patt or /pal given 6.7.99
*        *sh* 6/99 -- suppress color key for 2-argument version of command
* V510:  *sh* 3/00 -- change to how line specifications are passed

* V533 *acm*6/01 - add arguments is_logh, is_logv to PLOT_SET_UP call, 
*                  set log axes, inverse log axes and log depth axes with
*                  PLOT/VLOG or PLOT/HLOG
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES.
* v541 *acm*  5/02 add option to control outline on the shade key boxes 
*                   /KEY=continuous
* v554 *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
* v580 *acm* 6/15/04 In the workaround which does a PPL LEV command for data with
*                    a single value, need to give the PPL command the value with  
*                    more precision.
* V581 *acm* 5/05  fix bug 1249: /KEY=CONT. Need to use EQUAL_STR to get the
*                  argument to /KEY, then compare to see if it equals uppercase 'CON'
* V600 *acm* 5/06  with MODE LINECOLORS, number of lines not limited to num_fancy_lines
* V602 *acm* 11/06 Fix bug 1472: Where single-level color plot is requested, use levels (1) 
*                  setting rather than sending the data value; avoids precision problems.
* V6.11 4/08 *acm* Fixes for bug 1571; call to PPL_AXES_RESTORE moved to DISP_INIT
* V6.2  *acm* 11/08Fix bug 1609: If the plot is an overlay then do not process /AXES=
*                  or /NOAXES. Axes are never drawn for overlays anyway.
* V6.2  *acm* 2/09 Fix choice of color settings with /PEN and /THICK when we have 
*                  increased the num of colors with MODE linecolors
* V6.83 *acm*12/12 Add POLYGON/MODULO qualifier, for curvilienar coordinate plots, 
*                  as is done for SHADE and CONTOUR plots
* PyFr  *kms* 7/13 EQUAL_COLOR now returns a color ID, and EQUAL_THICK 
*                  returns a pen ID from a color ID and a thickness.
* PyFr  *kms* 7/13 Added /OPACITY
* V6.92 *acm* 6/14 New color /key options: any of /key=horizontal,centerlabel
* V695  *acm*  5/15 Ticket 2279: PyFerret don't redraw lines
*                   Do not draw axes on an overlay unless they
*                   ask for it with /AXES or a ppl axis command
* V697 *acm* 12/15 Fixes for bug 1237 move call of PPL_AXES_RESTORE back to end of plot routines
* v697 *acm* 12/15 Ticket 2331, add flag changed_key to common, to restore the color key
*                  style after a color plot and also after SHADE/SET; SHADE.
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* PyFr *kms*  4/17 Remove SAVE_FRAME for HDF format since no longer supported
* PyFr *acm*  1/18 Issue an error on use of /PATTERN
* v744 *acm* 11/18 Issue 1786: graticule lines & axes drawn if PLOT/OVER/AXES given
* v751 *acm*  5/19 Issue 1937: POLY/MODULO mis-handled requested /LEVELS
*       *acm* 3/22 Pyferret issue 51 Check the status flag after the call to SET_GRATICULES

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
	include 'slash.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'PPLDAT.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'cont_inc.decl' ! with iautoc and nlev
        include 'CONT.INC'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'    ! wsid

	
* calling argument declarations:

* internal variable declarations:
	LOGICAL     TM_HAS_STRING, no_range, do_key, spectrum, pattern,
     .			use_line, overlay, transpz, no_labels, set_up, 
     .			noaxes, axes, line_it, fill_it, is_logh, is_logv, 
     .			no_data, is_modx, do_mod_minus, do_mod_plus
	INTEGER         sp, patt, TM_LENSTR1, TM_LENSTR, STR_UPCASE
	INTEGER		symbol, icolor, ipen, status,
     .			loc, limit, slen, coord_ax,
     .			ax1_work, ax2_work,
     .			s1, s2, it_start(4), it_end(4),
     .			nparm,  num_it, i, iax(4), i1, loch, locv
	REAL		val, only_val, sym_size, add_lon
        REAL*4          thickness, opacity
	CHARACTER	LEFINT*4, TM_FMT*48, fname*128, poly_opt*10,
     .                  buff*25

* internal parameter declarations
	INTEGER ncolors
        LOGICAL continuous_key

* set up the graphics environment and get the data
	CALL START_PPLUS(.FALSE.)

	CALL DISP_DATA_SET_UP(  ax1_work, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* Initialize the lines buffer in PPL before setting up
	overlay   = qual_given( slash_shad_over_plot ).GT.0
	if (overlay) SOVER = .true.

	if (sover) then
CC      *jd* NLINES initializes SOVER, and resets MIN/MAX. Do not execute. 
CC       => Set values here instead.
	   LINEN=0
	   IBASE=1
	   PBUF=0
	   LNUM=1
	else
	   CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )
	endif

* initialize
	transpz   = qual_given( slash_shad_transpz   ).GT.0
	no_labels = qual_given( slash_shad_nolabels  ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0

* On overlays don't draw axes unless specificially requested with /AXES
* (e.g.so graticule lines can be overlaid on a final plot.)

	noaxes    = qual_given( slash_noaxes         ).GT.0

        IF (overlay) THEN
	   noaxes = .FALSE.
	   axes = .FALSE.
	ENDIF

	axes      = qual_given( slash_polygon_axes   ).GT.0

	coord_ax  = qual_given( slash_polygon_coord  )		! integer
	is_logh   = qual_given( slash_is_logh        ).GT.0
	is_logv   = qual_given( slash_is_logv        ).GT.0 

	symbol	  = qual_off	! or ??unspecified_int4??
	sym_size  = 0.08	! default plot mark size
        thickness = 1.0         ! default thickness
	icolor	  = unspecified_int4
        sp        = qual_given( slash_cont_spectrum  )

        spectrum  = sp .GT. 0 
        patt      = qual_given( slash_shad_pattern   )
        pattern   = patt .GT. 0
	IF (pattern) GOTO 5600  ! pattern not available in PyFerret
	add_lon	  = 0.

        ncolors = mode_arg(pmode_linecolors,1)

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* decide on line or fill or both
        use_line   = ( qual_given(slash_polygon_color) .GT. 0 ) .OR.
     .               ( qual_given(slash_polygon_thick) .GT. 0 ) .OR.
     .               ( qual_given(slash_polygon_line)  .GT. 0 )

	if (use_line) then
	   fill_it = qual_given(slash_polygon_fill) .gt. 0 
     .               .or. spectrum .or. pattern
	else
	   fill_it   = .true.
	end if

	if (use_line .and. fill_it) then
	   poly_opt = '/LINE/FILL'
	else if (use_line .and. .not. fill_it) then
	   poly_opt = '/LINE'
	else
	   poly_opt = ' '
	end if

* decode qualifiers
* ... POLYGON/COLOR=n
	loc = qual_given( slash_polygon_color )
	IF ( loc .GT. 0  ) THEN
*          get the color ID
           CALL EQUAL_COLOR(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                      icolor, thickness, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
        ELSEIF ( qual_given( slash_polygon_thick) .GT. 0 ) THEN
           icolor = 1  ! /THICK -> "BLACK" if color not given
	ENDIF

* ... CONTOUR/OPACITY=n
        loc = qual_given( slash_polygon_opacity )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_VAL(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    val, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
           IF ( (val .LT. 0.0) .OR. (val .GT. 100.0) ) GOTO 5500
           opacity = val / 100.0
*          Assume the opacity is for the spectrum colors, 
*          and not the pen, if filling is going to be done
           IF ( icolor .NE. unspecified_int4 .AND. .NOT. fill_it )
     .        CALL DUP_COLOR_SET_OPACITY(wsid, icolor, opacity)
           CALL SET_OVERRIDE_OPACITY(opacity)
        ELSE
*          overriding opacity not specififed
           opacity = -1.0
           CALL SET_OVERRIDE_OPACITY(opacity)
        ENDIF

* ... POLYGON/THICKNESS=n
 200	loc = qual_given( slash_polygon_thick )
	IF ( loc .GT. 0  ) THEN
*          From above, icolor will be specified
*          convert the color ID into a pen ID using the specified thickness
           CALL EQUAL_THICK(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                      thickness, icolor, ipen, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
        ELSEIF ( icolor .NE. unspecified_int4 ) THEN
*          convert the color ID into a pen ID using the default thickness
           CALL EQUAL_THICK(' ', thickness, icolor, ipen, status)
        ELSE
*          set ipen to unspecified_int4
           ipen = icolor
        ENDIF

* ... POLYGON/LINE_STYLE=n
	loc = qual_given( slash_polygon_line )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      ipen = NINT( val )	! user-specified line style
              limit = 3*mode_arg(pmode_linecolors,1)
	      IF ( ipen .LT. 1 .OR. ipen .GT. limit ) GOTO 5200
              IF ( opacity .GE. 0.0 )
     .           CALL DUP_STD_PEN_SET_OPACITY(wsid, ipen, opacity)
	   ELSEIF ( icolor .EQ. unspecified_int4 ) THEN
	      ipen = qual_on		! auto selection of line style
	   ENDIF
	ENDIF

* ... POLYGON/hgrat[=dash or line] default is dash
*     hgrat line type: 0=tics only, 1=dash 2=line

	loch = qual_given(slash_polygon_hgrat)

* ... POLYGON/vgrat[=dash or line] default is dash
*     vgrat line type: 0=tics only, 1=dash 2=line

	locv = qual_given(slash_polygon_vgrat)
        
        IF (loch .EQ. 0  .AND. locv .EQ. 0) THEN
           loch = qual_given(slash_polygon_grat)
           locv = qual_given(slash_polygon_grat)
        ENDIF

        CALL set_graticules(loch, locv, status)
	IF ( status .NE. ferr_ok ) GOTO 5000
	
* default key

	do_key =  (qual_given(slash_polygon_key)  .GT. 0)
     .          .OR. (fill_it
     .		     .AND. .NOT.overlay
     .		     .AND. num_uvars_in_cmnd.GT.2)
	IF (qual_given(slash_polygon_nokey) .gt. 0) do_key = .FALSE.
	

* Double check: is there no valid input data?
	IF ( no_data ) do_key = .FALSE.
	IF ( do_key ) THEN
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )

* ... /KEY[=continuous,horizontal,centerlabel ?
	      
	      loc = qual_given(slash_polygon_key )
	      changed_key = .FALSE.

              IF ( loc .GT. 0) THEN
		 CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
		 IF (TM_LENSTR(buff) .GT. 0) THEN
	         IF ( TM_HAS_STRING(buff, 'CON') ) THEN
                    changed_key = .TRUE.
                    CALL PPLCMD (from, line, 0, 'S_KEY 1', 1, 1 )
                 ENDIF
	         IF ( TM_HAS_STRING(buff, 'CEN') ) THEN
                    changed_key = .TRUE.
                    CALL PPLCMD (from, line, 0, 'S_KEY 2', 1, 1 )
                 ENDIF
	         IF ( TM_HAS_STRING(buff, 'HOR') ) THEN
                    changed_key = .TRUE.
                    CALL PPLCMD (from, line, 0, 'S_KEY 3', 1, 1 )
                 ENDIF
                 IF (.NOT. changed_key) GOTO 5460
		 ENDIF

              ENDIF

	ELSE
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
	ENDIF

* Do a /MODULO operation for curvilinear coordinates?

	is_modx = qual_given(slash_polygon_mod) .GT. 0

	if (overlay) sover = .true.
* set up the axes and title and load the data into PPLUS

 220	CONTINUE 
	CALL POLYGON_SET_UP   ( 
     .				overlay,
     .				transpz,
     .				coord_ax,
     .				no_labels,
     .				no_range,
     .				only_val,
     .				symbol,
     .				ipen,
     .				use_line,
     .				is_logh, 
     .				is_logv,
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				is_modx,
     .				do_mod_minus, 
     .				do_mod_plus,
     .				add_lon,
     .				workmem(ax1_work)%ptr,
     .				workmem(ax2_work)%ptr,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* is there no valid input data?
	no_data =  no_range .AND. only_val.EQ.bad_val4

* PPLUS blows up setting auto contour levels if all data have same value
* But, do not let this be rounded off, or we cannot find the color when matching 
* this value in get_ndx, called by fillpol

* The initial fix below had precision problems when the value in only_val was
* of high precision, sometimes causing polygons to go unfilled (bug 1472).  
* Instead, make a ppl lev 1 setting, as is done in disp_prep.F

c     IF ( no_range .AND. qual_given( slash_shad_lev_same ) .EQ. 0 )
c    .          CALL PPLCMD ( from, line, 0, 
c    .          'LEV () ('//TM_FMT(only_val, 8, 20, slen)//')' , 1, 1 )

      IF ( no_range .AND. qual_given( slash_shad_lev_same ) .EQ. 0 ) THEN
           iautoc = 1  
c           nlev = 1   
      ENDIF

* protect colors already on-screen before a color overlay
*        PyFerret does not need to protect colors after using them
*        IF ( overlay ) CALL PPL_SHASET( 'PROTECT' )

* save old and reset to new color spectrum if requested (if not doing
* a polygon/modulo operation).
	IF ( spectrum .AND. add_lon.EQ.0) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(sp):qual_end(sp)),
     .             fname, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( fname .EQ. ' ' ) fname = 'default'
	   CALL PPL_SHASET( 'SAVE' )
	   CALL PPL_SHASET( 'SPECTRUM='//fname(1:TM_LENSTR1(fname)) )
	ENDIF

* save old and set to new pattern if requested (a la spectrum)
	IF ( pattern ) THEN
	   CALL EQUAL_STR_LC( 
     .             cmnd_buff(qual_start(patt):qual_end(patt)),
     .             fname, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( fname .EQ. ' ' ) fname = 'default'
	   CALL PPL_PATSET( 'SAVE' )
	   CALL PPL_PATSET( 'PATTERN='//fname(1:TM_LENSTR1(fname)) )
	ENDIF

* turn off axes? 
	IF ( noaxes ) THEN 
           DO 300 i = 1, 4
              iax(i) = 0
 300       ENDDO
        ENDIF

* turn on (one or more) axes? (7/11/01)

* ... PLOT/AXES=(TOP,BOTTOM,LEFT,RIGHT)
	IF ( axes ) THEN
           loc = qual_given( slash_polygon_axes )
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000

	   IF ( buff .NE. ' ' ) THEN

* Get arguments, if any.  Otherwise turn on all axes
              slen = TM_LENSTR1(buff)
              s1 = 1
              s2 = slen
              IF (buff(1:1) .EQ. '(' )THEN
                  s1 = 2
                  s2 = slen-1
                  IF ( buff(slen:slen) .NE. ')' ) GOTO 5440
              ENDIF

              nparm = 4
              CALL parse_comma_list ( buff, s1, s2, nparm, 
     .             num_it, it_start, it_end, status)
              IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN	! /AXES=()
                 DO 400 i = 1, 4
                    iax(i) = 0
 400             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. nparm) THEN
                 GOTO 5440
              ELSE
                 DO 500 i = 1, nparm
                    READ (buff(it_start(i):it_end(i)), *) iax(i)
                    IF (iax(i) .LT. 0  .OR.  iax(i) .GT. 1) GOTO 5450
 500             CONTINUE
              ENDIF
           ELSE		! turn on all axes
	      DO 600 i = 1, 4
		 iax(i) = 1
 600	      CONTINUE
           ENDIF
              
	ENDIF

*  Turn off or on axes, including time axes.
        IF (axes .OR. noaxes) CALL ax_on_off (iax, overlay)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* polygon it 
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0,
     .          'FILLPOL/OVER/'//ppl_wait//poly_opt//' ' , 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0,
     .		'FILLPOL/'//ppl_wait//poly_opt//' ', 1, 1 )
	   ENDIF

* Is a modulo replication set up?  If so go do it, no labels, no axes, no key,
* same levels. If auto-levels were used on the first plot, re-using them is 
* fine. The levelsare computed with all the data. 

	   IF (is_modx) THEN
	      add_lon = 0.
	      IF (do_mod_minus) THEN
	         add_lon = -360.
		 do_mod_minus = .FALSE.
	      ELSEIF (do_mod_plus) THEN
	         add_lon = 360.
		 do_mod_plus = .FALSE.
	      ENDIF
	      IF (add_lon .NE. 0) THEN
		 sover = .TRUE.
		 overlay = .TRUE.
		 noaxes = .TRUE.
		 no_labels = .TRUE.
		 CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
		 iautoc = 0 ! reuse levels from last plot
C Reset pplus lines
		 LINEN=0
		 IBASE=1
		 PBUF=0
		 LNUM=1

		 GOTO 220
	      ENDIF
	   ENDIF

* protect the new colors and restore the original color spectrum
           IF ( spectrum ) THEN
*              PyFerret does not need to protect colors after using them
*              CALL PPL_SHASET( 'PROTECT' )
              CALL PPL_SHASET( 'SPECTRUM' )
           ENDIF

* restore the original pattern
           IF ( pattern ) THEN
              CALL PPL_PATSET( 'PATTERN' )
           ENDIF

* Mark the levels as set; if we used histogram levels it is 0

          IF (need_histo) iautoc = 1

* clear the overriding opacity
        opacity = -1.0
        CALL SET_OVERRIDE_OPACITY(opacity)

* restore axes? (5/29/97) revised 3/02
	IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* restore default (non-continuous) key style?
        IF (continuous_key ) CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

*  Save log axis state

	IF (overlay) THEN
	   is_logh = h_logaxis
	   is_logv = v_logaxis
	ELSE
	   h_logaxis= is_logh
	   v_logaxis = is_logv 
	ENDIF

* release the memory used for axis buffers
 1000 	CALL FREE_WS_DYNMEM( ax1_work )
	CALL FREE_WS_DYNMEM( ax2_work )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5200	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
 5400	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to 3', *1000 )
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', 
     .		*1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )
 5460	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Unrecognized argument to KEY command', *1000 )
 5500   CALL ERRMSG( ferr_out_of_range, status,
     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .          'Value from 0 to 100 must be given', *1000 )
 5600	CALL ERRMSG( ferr_invalid_command, status, 
     .		'PATTERN style is not available in PyFerret', *1000 )
	END
