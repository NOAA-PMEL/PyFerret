	SUBROUTINE XEQ_CONTOUR

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
* execute the CONTOUR command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/5/87 copied from rev. 0.0 of XEQ_PLOT (6/3/86) with changes
*		 to allow multiple mrs in a single expression
*		 and with separate subroutines for graphics set-up
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - incorporated 90 degree in-plane transpose
*			   and DISP_INIT separated from DISP_SET_UP
* revision 0.3 - 7/8/87  - DISP_LABELS separated from DISP_SET_UP
* revision 0.4 - 7/28/87 - added CALL DISP_PREP
* revision 0.5 - ????    - plots not automatically transposed in XT plane
* revision 0.6 - 3/2/88  - check for valid data range before PLOT+ auto-scaling
* revision 0.7 - 4/14/88 - added /NOLABELS option and range check-->DISP_SET_UP
* revision 0.8 - 4/28/88 - set up PPLUS LEV command if no data range
* revision 0.9 - 8/2/88  - added MODE WAIT
* revision 1.0 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* revision 1.1 - 1/25/89 - PPLUS interruptibility - chg for multiple plots
* revision 1.2 - 5/1/89  - PPL LEVELS not cleared for no_range if /LEVELS_SAME
* V200:  5/19/89 - 4D symmetrical
*		 - obtain oversized data region to fill contours to edges
*	  1/4/89 - using keys for labelling
*	 3/13/90 - check status from DISP_PREP
* V230:   6/8/92 - added CONTOUR/FILL/KEY/LINE/NOKEY
*       10/22/92 - removed dependence on shade_vars.cmn using PPLCMND(SHAKEY)
*        11/9/92 - protect colors before doing a color overlay
*                - default for KEY is off on overlays
* V300:  3/11/93 - added /SPECTRUM=
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*	 2/10/94 - added /PEN=
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V450:  5/29/97 - added /NOAXES qualifier (useful in curvilinear plots)
* V491:  12.98   - added /PATTERN qualifier *jd*
* V510 12/99 *sh* - allow plots of all-missing data
*		  - added /SIZE=/SPACING=/SIGDIG= controls over contour labels
*      5/00 *sh* - restored the /PEN qualifier (with "denigrated" remark
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
*      *acm*6/01 - add arguments is_logh, is_logv to DISP_SET_UP call.
*                  (only used for PLOT command)
*      *acm* 7/01 - Add /AXES=TOP,BOT,LEFT,RIGHT
*		    & call ax_on_off to implement /NOAXES.
*      *acm* 10/01  remove VMS includes
* V540 *acm*  2/02  restore persistence of PPL AXSET settings
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES
*      *acm*  4/02  do not set is_logh, is_logv.  They arent used by CONTOUR.
*      *acm*  4/02  If this is an overlay, and there is no valid data, 
*                   do not put "No Valid Data" across the middle of the plot.
*                   Instead (in disp_set_up) add this information to the
*                   variable name in the key below the plot.
* v541 *acm*  5/02 add option to control outline on the shade key boxes 
*                   /KEY=continuous
* V550: *acm*11/02 add argument is_shade to DISP_SET_UP call (not used by contour)
* V552: *acm* 5/03 add argument use_cell to DISP_SET_UP call (not used by contour)
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
* v580 *acm* 6/15/04 In the workaround which does a PPL LEV command for data with
*                    a single value, need to give the PPL command the value with  
*                    more precision. (single value data isnt contoured or filled,
*                    but put this fix in as a parallel to what is done in shade
*                    and polygon.)
* V581 *acm* 5/05  fix bug 1249: /KEY=CONT. Need to use EQUAL_STR to get the
*                  argument to /KEY, then compare to see if it equals uppercase 'CON'
* V602 *acm* 11/06 Fix bug 1472: Where single-level color plot is requested, use levels (1) 
*                  setting rather than sending the data value; avoids precision problems.
* V603 *acm*  5/07 as in xeq_shade, for fix of bug 904: if FILL/LINE, then set iautoc and 
*                  NLEV2 to re-use the contour levels from the shade plot. The bug does not
*                  seem to occur on fill plots, but might as well not recompute levels.
* V6.11 4/08 *acm* Fixes for bug 1571; call to PPL_AXES_RESTORE moved to DISP_INIT
* V614  *acm* 10/08 - add new MODE NODATA_LAB to control the NO VALID DATA label.
*                  Put it on plots only when this mode is set.
* V6.2  *acm* 11/08Fix bug 1609: If the plot is an overlay then do not process /AXES=
*                  or /NOAXES. Axes are never drawn for overlays anyway.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*                  A time axis may be a T axis or F axis
* V6.85 *acm* 3/13 Fix ticket 2049; check we can open palette file
* PyFr  *kms*  6/13 EQUAL_COLOR now returns a color ID, and EQUAL_THICK 
*                   returns a pen ID from a color ID and a thickness.
* PyFr  *kms*  6/13 Added /THICK /OPACITY
* PyFr  *kms*  7/13 Scale default lettering sizes by textscale;
*                   use scaled label heights and spacings instead of 
*                   hard-coded values.
* V6.92 *acm* 6/14 New color /key options: any of /key=horizontal,centerlabel
* v695  *acm* 5/16 When running PyFerret do not draw axes on an overlay unless they
*                  ask for it with /AXES or a ppl axis command
* V697 *acm* 12/15 Fixes for bug 2327 move call of PPL_AXES_RESTORE back to end of plot routines
* v697 *acm* 12/15 Ticket 2331, add flag changed_key to common, to restore the color key
*                  style after a color plot and also after SHADE/SET; SHADE.
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* PyFr *kms*  4/17 Remove SAVE_FRAME for HDF format since no longer supported
* PyFr *acm*  1/18 Issue an error on use of /PATTERN
* V74  1/18 *am* for issue #1854, extra argument to DISP_SET_UP
* V741 *acm*  5/18 issue 1874: consistently apply /CONSET number of digits setting
* v744 *acm* 11/18 Issue 1786: graticule lines & axes drawn if PLOT/OVER/AXES given
* v765 *acm* 11/21 PyFerret Issue 99: Change the color of contour labels on CONTOUR/COLOR=
*                  when using PyFerret fonts

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
        include 'parampl5_dat.decl'     ! with inclen
        include 'PARAMPL5.DAT'  ! with inclen
        include 'cont_inc.decl' ! with iautoc and nlev
        include 'CONT.INC'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'    ! wsid
        include 'pyfonts.cmn'

	
* calling argument declarations:

* internal variable declarations:
	LOGICAL	TM_HAS_STRING, 
     .            no_range, overlay, transpz, no_labels, fill, usa_line,
     .            do_key, spectrum, set_up, noaxes, axes, pattern, 
     .		  no_data, is_logh, is_logv, inc_grid
	INTEGER	  TM_LENSTR, TM_LENSTR1, status, slen, sp, ipen, icolor, 
     .		  ax1_work, ax2_work, loc, sigdig,
     .		  patt, i, nparm, num_it, 
     .		  s1, s2, it_start(4), it_end(4), iax(4), density,
     .            loch, locv, igrp
        INTEGER   STR_UPCASE, STR_SAME, i1
	REAL	  vbuf, only_val, aspect, lab_size, lab_spacing
        REAL*4    dfltthick, opacity
	PARAMETER (aspect = 0.0)			!kob 12/96
	CHARACTER TM_FMT*48, TM_INQ_PATH*2048, 
     .                  fname*128, buff*25, sfile*2048

* internal parameter declarations:
	LOGICAL		norm_dims_only, not_vector, 
     .                  is_shade, use_cell
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			not_vector	= .FALSE. )

* set up the graphics environment

	CALL START_PPLUS(.FALSE.)

* get the data (possibly ?? on a slightly oversized region)
!	cx_ovsiz(cx_last) = ovsiz_plane	! ... causes too many headaches !
	CALL DISP_DATA_SET_UP(  ax1_work, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* initialize
	overlay   = qual_given( slash_cont_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_cont_nolabels  ).GT.0
	transpz   = qual_given( slash_cont_transpz   ).GT.0
	set_up    = qual_given( slash_cont_set_up    ).GT.0
	fill      = qual_given( slash_cont_fill      ).GT.0

* On overlays don't draw axes unless specificially requested with /AXES
* (e.g.so graticule lines can be overlaid on a final plot.)

	noaxes    = qual_given( slash_noaxes         ).GT.0

        IF (overlay) THEN
	   noaxes = .FALSE.
	   axes = .FALSE.
	ENDIF

	axes      = qual_given( slash_cont_axes      ).GT.0
        
	sp        = qual_given( slash_cont_spectrum  )

        spectrum  = sp .GT. 0 .AND. fill
        is_shade  = .FALSE.
        use_cell  = .FALSE.
	
	is_logh   = .FALSE.	  ! Not valid quals
	is_logv   = .FALSE.	  ! for contour

        patt     = qual_given( slash_cont_pattern  )
        pattern  = patt .GT. 0 .AND. fill
	IF (pattern) GOTO 5600  ! pattern not available in PyFerret

	usa_line  = .NOT.fill
     .          .OR. qual_given( slash_cont_line      ).GT.0
	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,
     .                                       num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( (plot_axis(2).EQ.t_dim .OR. plot_axis(2).EQ.f_dim) .AND. 
     .		plot_axis(1).NE.x_dim ) transpz = .NOT.transpz

* ... CONTOUR/SIZE=float
	loc = qual_given( slash_cont_size )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf.LE. 0.0 ) GOTO 5210
	      lab_size = vbuf	
	   ENDIF
	ELSE
	   lab_size = wn_con_hgt(wsid)
	ENDIF
* ... CONTOUR/SPACING=float
	loc = qual_given( slash_cont_spacing )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf .LE. 0.0 ) GOTO 5210
	      lab_spacing = vbuf	
	   ENDIF
	ELSE
	   lab_spacing = wn_con_labsp(wsid)
	ENDIF
* ... CONTOUR/SIGDIG=integer
	loc = qual_given( slash_cont_sigdig )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf.LE.0.5 ) GOTO 5210
	      sigdig = NINT(vbuf)
	   ENDIF
	ELSE
	   sigdig = 0
	ENDIF
* ... CONTOUR/COLOR=color name or pen number
        dfltthick = 1.0
	loc = MAX( qual_given(slash_pen), qual_given(slash_color))
	IF ( loc .GT. 0 ) THEN
	   IF (.NOT.denig_pen_msg_done .AND. qual_given(slash_pen).GT.0)
     .									THEN
	      CALL WARN( '/PEN is deprecated. Use /COLOR.')
	      denig_pen_msg_done = .TRUE.
	   ENDIF
*          get the color ID and default thickness
           CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                       icolor, dfltthick, status )
           IF ( status .NE. ferr_ok ) GOTO 5000

* If we're using PyFerret fonts and no contour-specific font settings have been made, make the
* contour label color match the contour-line color.

           contour_color_pyf = .FALSE.

           IF (STR_SAME(pyfnt_fontname, 'hershey') .NE. 0) THEN
	       igrp = 8  ! contour labels text-group

	       CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
	       IF (.NOT.pyf_set(igrp)) THEN
	       
	           CALL PARSE_PYFONT_COLOR (buff, pyf_red(igrp), 
     .                        pyf_grn(igrp), pyf_blu(igrp), status)
	           
	           pyf_italic(igrp) = pyfnt_italic
	           pyf_bold(igrp) = pyfnt_bold
	           pyf_isiz(igrp) = pyfnt_isiz
	           pyf_fontname(igrp) = pyfnt_fontname

		   pyf_set(igrp) = .TRUE.

		   contour_color_pyf = .TRUE.

	       ENDIF
		 
           ENDIF

	ELSE
	   icolor = -1	! auto-select in DISP_SET_UP
	ENDIF

* ... CONTOUR/OPACITY=n
        loc = qual_given( slash_cont_opacity )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_VAL(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    vbuf, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
           IF ( (vbuf .LT. 0.0) .OR. (vbuf .GT. 100.0) ) GOTO 5500
           opacity = vbuf / 100.0
*          Assume the opacity is for the spectrum colors, 
*          and not the pen, if fill is specified
           IF ( icolor .GE. 0 .AND. .NOT. fill )
     .        CALL DUP_COLOR_SET_OPACITY(wsid, icolor, opacity)
           CALL SET_OVERRIDE_OPACITY(opacity)
        ELSE
*          overriding opacity not specififed
           opacity = -1.0
           CALL SET_OVERRIDE_OPACITY(opacity)
        ENDIF

* ... CONTOUR/THICKNESS=n
        loc = qual_given( slash_cont_thick )
        IF ( loc .GT. 0  ) THEN
           IF ( icolor .GE. 0 ) THEN
              CALL EQUAL_THICK(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                         dfltthick, icolor, ipen, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ELSE
*             leave as the auto-select value (-1)
              CALL WARN ('/THICKNESS ignored unless /COLOR (or /PEN) given')
              ipen = icolor
           ENDIF
        ELSEIF ( icolor .GE. 0 ) THEN
*          /THICK not given; convert color ID to pen ID using default thickness
           CALL EQUAL_THICK(' ', dfltthick, icolor, ipen, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
        ELSE
*          /THICK and /COLOR not given; leave as auto-select value (-1)
           ipen = icolor
        ENDIF

* ... CONTOUR/hgrat[=dash or line] default is dash
*     hgrat line type: 0=tics only, 1=dash, 2=line

	loch = qual_given(slash_cont_hgrat)

* ... CONTOUR/vgrat[=dash or line] default is dash
*     vgrat line type: 0=tics only, 1=dash, 2=line

	locv = qual_given(slash_cont_vgrat)

        IF (loch .EQ. 0  .AND. locv .EQ. 0) THEN
           loch = qual_given(slash_cont_grat)
           locv = qual_given(slash_cont_grat)
        ENDIF

        CALL set_graticules(loch, locv, status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* ... /DEGMINSEC[=1 or 2] Request deg/min or deg/min/sec axis labels
	loc = qual_given( slash_cont_dms )
        dms = 0
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
           dms = 1
           IF (vbuf .GT. 0) THEN
	      dms = vbuf
	      IF (dms .GT. 2) GOTO 5470
	   ENDIF
	ENDIF

* ... /HALFSPACE Request deg/min or deg/min/sec axis labels
	loc = qual_given( slash_cont_halfsp )
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
           lonlatspace = 1
           IF (vbuf .GE. 0) lonlatspace = vbuf
	ENDIF

* set contour label details (3/00)
* fix bug 
	IF (sigdig .NE. 0) THEN
	   WRITE ( ppl_buff, 3100 ) lab_size, sigdig, lab_spacing
	ELSE
	   WRITE ( ppl_buff, 3200 ) lab_size, lab_spacing
	ENDIF
	   
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3100	FORMAT('CONSET ',F8.3,',',I4,', , , , , ,',F7.2)
 3200	FORMAT('CONSET ',F8.3,', , , , , , ,',F7.2)

	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

        density = 0	! not used; vec/flow needs it in set up

* set up the axes and title and load the data into PPLUS
	inc_grid = set_up
	CALL DISP_SET_UP   ( 	
     . 			 	not_vector,
     .				is_shade,
     .				use_cell,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_range,
     .				only_val,
     .                          density,
     .				is_logh, 
     .				is_logv,
     .				is_mr,		! pass list
     .				num_uvars_in_cmnd,
     .				is_cx,		! pass list
     .				is_uvar,	
     .				workmem(ax1_work)%ptr,
     .				workmem(ax2_work)%ptr,
     .				ipen,
     .				inc_grid,
     .				set_up,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* increment counter of line-style contours on screen
	IF ( usa_line ) ncont_on = ncont_on + 1

* is the no valid input data?
        no_data =  no_range .AND. only_val.EQ.bad_val4


* PPLUS blows up setting auto contour levels if all data have same value
* But, do not let this be rounded off, or we cannot find the color when matching 
* this value in get_ndx, called by fillpol.  ( although, if there is just
* one level, the fill is not done.)

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

* default key
	   do_key =  (qual_given(slash_cont_key)  .GT. 0)
     .		.OR. (.NOT.usa_line .AND. .NOT.overlay)
	   IF ( no_data ) do_key = .FALSE.
	   IF (qual_given(slash_cont_nokey) .GT. 0) do_key = .FALSE.
           IF ( do_key ) THEN
              CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )

* ... /KEY[=continuous,horizontal,centerlabel ?

              loc = qual_given( slash_shad_key )
	      changed_key = .FALSE.

              IF ( loc .GT. 0) THEN
		 CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
		 
		 IF (TM_LENSTR(buff) .GT. 0) THEN
	         changed_key = .FALSE.
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

c when it's an overlay, 'No Valid Data' is part of the plot key,
c done in disp_set_up. If mode_nodata_lab is false then leave this off.

        IF (.NOT. overlay .AND. no_data .AND. mode_nodata_lab) 
     .                  CALL PUT_LABEL( ppl_movlab, 0.5, 0.5,
     .                  ppl_centered, dflt_letsize_title*textscale,
     .                  'No Valid Data',
     .                  lab_loc_axlen_times, lab_loc_axlen_times )
           ENDIF

* protect colors already on-screen before a color overlay
*              PyFerret does not need to protect colors after using them
*           IF ( overlay .AND. fill ) CALL PPL_SHASET( 'PROTECT' )

* save old and reset to new color spectrum if requested
           IF ( spectrum ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(sp):qual_end(sp)),
     .             fname, status )
              IF ( status .NE. ferr_ok ) RETURN
              IF ( fname .EQ. ' ' ) fname = 'default'
	      
	      slen = TM_LENSTR1(fname)
              sfile = TM_INQ_PATH (fname(:slen),'FER_PALETTE','.spk',
     .        	 .false.,status)
              IF (status .NE. ferr_ok) go to 5300 

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
	      
	      slen = TM_LENSTR1(fname)
              sfile = TM_INQ_PATH (fname(:slen),'FER_PALETTE','.pat',
     .        	 .false.,status)
              IF (status .NE. ferr_ok) go to 5300 

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
           loc = qual_given( slash_cont_axes )
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

* contour it 
        IF ( fill ) THEN
* ... filled contours
	   IF ( overlay ) THEN
	      IF ( usa_line ) THEN
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/OVER/' //ppl_wait//' ' , 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
	         iautoc = 0 ! reuse levels from last plot
	         nlev = nlev2
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      ELSE
	         CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/OVER/' //ppl_wait//' ' , 1, 1 )
	      ENDIF
	   ELSE
	      IF ( usa_line ) THEN
	      
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/LATCH/'//ppl_wait//' ', 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
	         iautoc = 0 ! reuse levels from last plot
	         nlev = nlev2
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      ELSE
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/LATCH/'//ppl_wait//' ', 1, 1 )
	      ENDIF
	   ENDIF
        ELSE
* ... line-style contour

	   IF (contour_fewer .EQ. 1) THEN
	      iautoc = 1
	      nlev = 10
	   ENDIF
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/LATCH/'//ppl_wait//' ', 1, 1 )
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

* restore axes? (5/29/97  revised 3/2002 

        IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE
     
* If we auto-matched contour labels with contour color, unset that.
        IF ( contour_color_pyf ) pyf_set(igrp) = .FALSE.

* restore default key style?
        IF (changed_key ) CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* release the memory used for axis buffers
 1000 	CALL FREE_WS_DYNMEM( ax1_work )
	CALL FREE_WS_DYNMEM( ax2_work )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
! 5200   CALL ERRMSG( ferr_out_of_range, status,
!     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
!     .          'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
 5210	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Value must be positive', *1000 )
 5300   CALL ERRMSG (ferr_file_not_found, status, 
     .          'unable to open "' //
     .	fname(:slen)//'.spk" to set plot colors or pattern.', *1000)
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', *1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )
 5460	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Unrecognized argument to KEY command', *1000 )
 5470	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Argument to DegMinSec qualifier must be 1 or 2', *1000 )
 5500   CALL ERRMSG( ferr_out_of_range, status,
     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .          'Value from 0 to 100 must be given', *1000 )
 5600	CALL ERRMSG( ferr_invalid_command, status, 
     .		'PATTERN style is not available in PyFerret', *1000 )

	END
