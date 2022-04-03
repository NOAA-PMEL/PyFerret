	SUBROUTINE XEQ_VECTOR

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
* execute the VECTOR command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer uvnder VMS operating system
*
* revision 0.0 - 1/7/87
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - provisions to permit 90 degree data-plane transpose
*			   and to separate DISP_INIT from DISP_SET_UP
* revision 0.3 - 7/8/87  - DISP_LABELS separated from DISP_SET_UP
* revision 0.4 - 7/28/87 - added CALL DISP_PREP
* revision 0.5 - 8/11/87 - VECKEY relocates according to title position
* revision 0.6 - 8/26/87 - dont transpose on XT plots
* revision 0.7 - 3/2/88  - check for valid data range before loading
* revision 0.8 - 4/14/88 - added /NOLABELS and range check --> DISP_SET_UP
* revision 0.9 - 4/28/88 - added only_x to DISP_SET_UP
* revision 1.0 - 6/22/88 - compute axis aspect correction using PPLUS AXIS.INC
*			   axis lengths
* revision 1.1 - 8/1/88  - corrected bug: x_skip said "Y" and visa versa
*			   added MODE WAIT
* revision 1.2 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* V200:  5/19/89 - 4D symmetrical
*	  1/4/89 - using keys for labelling
*	 3/13/90 - check status from DISP_PREP
* V301: 12/8/93  - added VECTOR/XSKIP/YSKIP
* 	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*	 2/10/94 - added /PEN=
* V312: 5/94 - array "memory" as a calling argument
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* V450:  5/29/97 - added /NOAXES qualifier (eventually for curvilinear plots)
* V510 *sh* 12/99 - allow plots of all-missing data
*		  - replace /PEN qualifier with /COLOR, supporting named colors
*      5/00 *sh* - restored the /PEN qualifier (with "denigrated" remark
*      10/00 *AM* - added VECTOR/FLOWLINE (alias FLOWLINE)
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
*      *acm*6/01 - add arguments is_logh, is_logv to DISP_SET_UP call.
*                  (only used for PLOT command)
*      *acm* 7/01 - Add VECTOR/AXES=TOP,BOT,LEFT,RIGHT
*      *acm* 7/01 - Add /AXES=TOP,BOT,LEFT,RIGHT
*		    & call ax_on_off to implement /NOAXES.
*      *acm* 10/01  remove VMS includes
* V540 *acm*  2/02  restore persistence of PPL AXSET settings
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES
*      *acm*  4/02  If this is an overlay, and there is no valid data, 
*                   do not put "No Valid Data" across the middle of the plot.
*                   Instead (in disp_set_up) add this information to the
*                   variable name in the key below the plot.
*      *acm*  5/02  Add /NOKEY option.  Also, if /NOLAB specifed, do not
*                   plot a key.
* V550: *acm*11/02 add argument is_shade to DISP_SET_UP call (not used by vector)
* V552: *acm* 5/03 add argument use_cell to DISP_SET_UP call (not used by vector)
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V6.11 4/08 *acm* Fixes for bug 1571; call to PPL_AXES_RESTORE moved to DISP_INIT
* V614  *acm* 10/08 - add new MODE NODATA_LAB to control the NO VALID DATA label.
*                  Put it on plots only when this mode is set.
* V6.2  *acm* 11/08Fix bug 1609: If the plot is an overlay then do not process /AXES=
*                  or /NOAXES. Axes are never drawn for overlays anyway.
* V6.2  *acm& 2/09 Fix choice of color settings with /PEN and /THICK when we have 
*                  increased the num of colors with MODE linecolors
* V6.63 *acm* 7/10 Save symbols PPL_VEC_XSKIP, PPL_VEC_YSKIP containing the 
*                  Vector XSKIP and YSKIP values.
* V6.63 *acm* 7/10 Save symbols PPL_VEC_XSKIP, PPL_VEC_YSKIP containing the 
*                  Vector XSKIP and YSKIP values. (PPL_VECLEN is saved in vecfld)
* V6.63 *acm* 7/10 Add VECTOR/KEY to force key even if /NOLAB
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*                  A time axis may be a T axis or F axis
* PyFr  *kms*  6/13 EQUAL_COLOR now returns a color ID, and EQUAL_THICK 
*                   returns a pen ID from a color ID and a thickness.
* PyFr  *kms*  6/13 Added /OPACITY
* PyFr  *kms*  7/13 Scale default lettering sizes by textscale
* V695  *acm*  5/15 Ticket 2279: PyFerret don't redraw lines
*                   Do not draw axes on an overlay unless they
*                   ask for it with /AXES or a ppl axis command
* V697 *acm* 12/15 Fixes for bug 1237 move call of PPL_AXES_RESTORE back to end of plot routines
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* PyFr *kms*  4/17 Remove SAVE_FRAME for HDF format since no longer supported
* V74  1/18 *am* for issue #1854, extra argument to DISP_SET_UP
* v744 *acm* 11/18 Issue 1786: graticule lines & axes drawn if PLOT/OVER/AXES given
*       *acm* 3/22 Pyferret issue 51 Check the status flag after the call to SET_GRATICULES

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include	'plot_setup.parm'
	include	'slash.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
	include 'xpplus.cmn'
        include 'axis_inc.decl'       ! with axis lengths	
        include 'AXIS.INC'       ! with axis lengths	
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'    ! wsid
        include 'xrevision.cmn'

* calling argument declarations:

* internal variable declarations:
	LOGICAL		overlay, transpz, no_labels, no_x_rng,
     .			set_up, noaxes, axes, no_data, flows, 
     .			is_logh, is_logv, no_key, yes_key,
     .			is_shade, use_cell, inc_grid
	INTEGER		TM_LENSTR1, status, cx, 
     .			loc, icolor, ipen,
     .			dim1, dim2, asp,
     .			ax1_work, ax2_work,
     .			x_skip, y_skip,
     .                  density, slen, i, nparm, num_it, 
     .			s1, s2, it_start(4), it_end(4), iax(4),
     .                  loch, locv, ier, ist
        INTEGER   STR_UPCASE, i1
	REAL		only_x, aspect, vbuf, xloc, yloc
        REAL*4          thickness, opacity
	CHARACTER	buff*25


* internal parameter declarations:
	LOGICAL		norm_dims_only, vector
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			vector		= .TRUE.)
	INTEGER ncolors

* set up the graphics environment and get the data
	CALL START_PPLUS(.FALSE.)
	CALL DISP_DATA_SET_UP(  ax1_work, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* make sure the number of expressions given is even ( x and y components )
	IF ( MOD( num_uvars_in_cmnd, 2 ) .NE. 0 ) CALL ERRMSG
     .		(ferr_invalid_command,status,
     .		 'VECTOR requires paired components',*5000)

* initialize
	overlay   = qual_given( slash_vect_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_vect_nolabels  ).GT.0
	transpz   = qual_given( slash_vect_transpz   ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0

* On overlays don't draw axes unless specificially requested with /AXES
* (e.g.so graticule lines can be overlaid on a final plot.)

	noaxes    = qual_given( slash_noaxes         ).GT.0

        IF (overlay) THEN
	   noaxes = .FALSE.
	   axes = .FALSE.
	ENDIF

	axes      = qual_given( slash_vect_axes      ).GT.0

	flows     = qual_given( slash_vect_flowline  ).GT.0
	is_logh   = .FALSE.  ! Not valid quals
	is_logv   = .FALSE.  ! for vector

	no_key = .FALSE.
        IF (no_labels) no_key = .TRUE.

	IF (qual_given( slash_vect_nokey ).GT.0)  no_key    = .TRUE.

	! for vector, key even if /NOLAB
	IF (qual_given( slash_vect_key) .GT.0) THEN
	   no_key    = .FALSE.
	   yes_key   = .TRUE.  ! to override settings where no key would be drawn.
	ENDIF

        is_shade  = .FALSE.
        use_cell  = .FALSE.
        ncolors = mode_arg(pmode_linecolors,1)

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
* ... default orientation for YT and ZT VECTORs has time on horizontal
	IF ( (plot_axis(2).EQ.t_dim .OR. plot_axis(2).EQ.f_dim) .AND. 
     .		plot_axis(1).NE.x_dim ) transpz = .NOT.transpz

	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* possibly transpose the axes
	IF ( transpz ) THEN
	   dim2 = plot_axis(1)
	   dim1 = plot_axis(2)
	ELSE
	   dim1	= plot_axis(1)
	   dim2	= plot_axis(2)
	ENDIF

* scale-ratio correction requested ?
	IF ( qual_given( slash_vect_aspect ) .GT. 0 ) THEN
	   asp = qual_given( slash_vect_aspect )
	   CALL EQUAL_VAL( cmnd_buff(qual_start(asp):qual_end(asp)),
     .			   aspect, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( aspect .EQ. unspecified_val4 ) aspect = ylen/xlen
	   IF ( aspect .LE. 0.0 ) CALL ERRMSG( ferr_out_of_range,
     .		status, cmnd_buff(qual_start(asp):qual_end(asp)), *5000 )
	ELSE
	   aspect = 0.0
	ENDIF

* user-specified pen color ?  (2/94)
        thickness = 1.0
	loc = MAX( qual_given(slash_pen), qual_given(slash_color))
	IF ( loc .GT. 0 ) THEN
	   IF (.NOT.denig_pen_msg_done .AND. qual_given(slash_pen).GT.0)
     .									THEN
	      CALL WARN( '/PEN is deprecated. Use /COLOR.')
	      denig_pen_msg_done = .TRUE.
	   ENDIF
*          get the color ID
           CALL EQUAL_COLOR(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                      icolor, thickness, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
        ELSE IF ( qual_given( slash_vect_thick ) .GT. 0 ) THEN
           icolor = 1  ! /THICK alone means "BLACK" if color not specified
        ELSE
           icolor = -1  ! auto-select in DISP_SET_UP
	ENDIF

* ... CONTOUR/OPACITY=n
        loc = qual_given( slash_vect_opacity )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_VAL(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    vbuf, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
           IF ( (vbuf .LT. 0.0) .OR. (vbuf .GT. 100.0) ) GOTO 5500
           opacity = vbuf / 100.0
           IF ( icolor .GE. 0 )
     .        CALL DUP_COLOR_SET_OPACITY(wsid, icolor, opacity)
           CALL SET_OVERRIDE_OPACITY(opacity)
        ELSE
*          overriding opacity not specififed
           opacity = -1.0
           CALL SET_OVERRIDE_OPACITY(opacity)
        ENDIF

* ... VECTOR/THICKNESS=n
	loc = qual_given( slash_vect_thick )
	IF ( loc .GT. 0  ) THEN
*          From above, icolor will be specified
*          convert the color ID into a pen ID using the specified thickness
           CALL EQUAL_THICK(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                      thickness, icolor, ipen, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
        ELSEIF ( icolor .GE. 0 ) THEN
*          convert the color ID into a pen ID using the default thickness
           CALL EQUAL_THICK(' ', thickness, icolor, ipen, status)
        ELSE
*          set ipen to auto-select
           ipen = icolor
        ENDIF

* for flowline plots, get the density
        density = 0
        IF (flows) THEN 
	   loc = qual_given( slash_vect_density )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         density = 1.
	      ELSE
	         density = ABS(vbuf)
	      ENDIF
           ELSE
              density = 5
           ENDIF
        ENDIF

* ... VECTOR/hgrat[=dash or line] default is dash
*     hgrat line type: 0=tics only, 1=dash 2=line

	loch = qual_given(slash_vect_hgrat)

* ... VECTOR/vgrat[=dash or line] default is dash
*     vgrat line type: 0=tics only, 1=dash 2=line

	locv = qual_given(slash_vect_vgrat)
        
        IF (loch .EQ. 0  .AND. locv .EQ. 0) THEN
           loch = qual_given(slash_vect_grat)
           locv = qual_given(slash_vect_grat)
        ENDIF

        CALL set_graticules(loch, locv, status)
        IF ( status .NE. ferr_ok ) GOTO 5000

* set up title and load the x component data into PPLUS

	inc_grid = .FALSE.
	CALL DISP_SET_UP   ( 	
     . 			 	vector,
     .				is_shade,
     .				use_cell,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_x_rng,
     .				only_x,
     .                          density,
     .				is_logh, 
     .				is_logv,
     .				is_mr,
     .				num_uvars_in_cmnd,
     .				is_cx,
     .				is_uvar,	
     .				workmem(ax1_work)%ptr,
     .				workmem(ax2_work)%ptr,
     .				ipen,
     .				inc_grid,
     .				set_up,
     .				status )

	IF ( status .NE. ferr_ok ) GOTO 5000

* is there no valid input data?
        no_data =  no_x_rng .AND. only_x.EQ.bad_val4

c when it's an overlay, 'No Valid Data' is part of the plot key,
c done in disp_set_up. If mode_nodata_lab is false then leave this off.

        IF (.NOT. overlay .AND. no_data .AND. mode_nodata_lab) 
     .                  CALL PUT_LABEL( ppl_movlab, 0.5, 0.5,
     .                  ppl_centered, dflt_letsize_title*textscale,
     .                  'No Valid Data',
     .                  lab_loc_axlen_times, lab_loc_axlen_times )

* increment counter of vector fields on the screen
	nvect_on = nvect_on + 1

* special set-up required only for the VECTOR command
        IF (no_data) no_key  = .TRUE.  ! regardless of /KEY or /NOKEY setting

         IF ( .NOT.overlay .AND. .NOT.no_data .AND. .NOT.no_key) THEN
* ... vector length key below plot title (overlay written as key)
            xloc = (xlen/2.) - 0.5
            yloc = -1.*yorg + 0.2
            WRITE ( ppl_buff, '(2F6.1)' ) xloc, yloc
            CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )
         ENDIF   

	IF ( overlay .AND. yes_key ) THEN  ! force key in the middle under the title.
* ... vector length key below plot title (overlay written as key)
           xloc = (xlen/2.) - 0.5
           yloc = -1.*yorg + 0.2
	   WRITE ( ppl_buff, '(2F6.1)' ) xloc, yloc
	   CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )
	ENDIF	

	IF (no_key) THEN
* ... no vector-length key.  
	   CALL PPLCMD (' ',' ',0,'VECKEY 0', 1, 1 )
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
           loc = qual_given( slash_vect_axes )
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

* done now if /set_up, only
	IF ( set_up) GOTO 1000

* sub-sample the vectors to make a cleaner plot ?
* *ACM 10/00*  do not subsample if its a flowline plot

        IF (.NOT. flows) THEN
	   cx   = is_cx( 1 )
	   loc = qual_given( slash_vect_xskip )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         x_skip = 1.
	      ELSE
	         x_skip = ABS(vbuf)
	      ENDIF
	   ELSE
	      x_skip = ( cx_hi_ss(cx,dim1)-cx_lo_ss(cx,dim1) )
     .		 / max_x_vec + 1
	      IF ( x_skip .NE. 1 ) THEN
	         risc_buff = ' '	         
	         WRITE ( risc_buff, 3000 ) x_skip, 'X'
	         CALL SPLIT_LIST(pttmode_explct, err_lun, risc_buff, 0)
	      ENDIF
	   ENDIF
	   loc = qual_given( slash_vect_yskip )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         y_skip = 1.
	      ELSE
	         y_skip = ABS(vbuf)
	      ENDIF
	   ELSE
	      y_skip = ( cx_hi_ss(cx,dim2)-cx_lo_ss(cx,dim2) )
     .	   	    / max_y_vec + 1
	      IF ( y_skip .NE. 1 ) THEN 
	         risc_buff = ' '	         
	         WRITE ( risc_buff, 3000 ) y_skip, 'Y'
	         CALL SPLIT_LIST(pttmode_explct, err_lun, risc_buff, 0)
	      ENDIF
	   ENDIF
 3000	   FORMAT (' Using every ',I3,'th vector in the ',
     .            A1,' direction')
	   WRITE ( ppl_buff, '(2I4)' ) x_skip, y_skip

* plot it with vector arrows
	   IF (  overlay ) THEN
	      CALL PPLCMD (' ',' ',0,
     .		'VECTOR/OVER/' //ppl_wait//' '//ppl_buff, 1, 1 )
	   ELSE
	      CALL PPLCMD (' ',' ',0,
     .		'VECTOR/LATCH/'//ppl_wait//' '//ppl_buff, 1, 1 )
	   ENDIF

C   Save the skip values in a symbol PPL_VEC_XSKIP, PPL_VEC_YSKIP

	   WRITE ( ppl_buff, 3010 ) x_skip
 3010	   FORMAT ('PPL_VEC_XSKIP', i4)
	   ier = 0
	   ist = 0
	   CALL SETSYM (ppl_buff, 17, ier, ist)

	   WRITE ( ppl_buff, 3020 ) y_skip
 3020	   FORMAT ('PPL_VEC_YSKIP', i4)
	   ier = 0
	   ist = 0
	   CALL SETSYM (ppl_buff, 17, ier, ist)

        ELSE  

* plot it with flowlines

           CALL flow_set_up (is_mr, is_cx, transpz)

* special set-up required only for the VECTOR/FLOW command
	IF ( .NOT.overlay .AND. .NOT.no_data ) THEN
* ... vector length key below plot title (overlay written as key)
           xloc = (xlen/2.) - 0.5
           yloc = -1.*yorg + 0.2
	   WRITE ( ppl_buff, '(2F6.1)' ) xloc, yloc
	   CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )

           IF (no_key) THEN
	      CALL PPLCMD (' ',' ',0,'VECKEY 0', 1, 1 )
	   ENDIF	

	ENDIF	

           WRITE (ppl_buff, '(I4)') density
	   IF (  overlay ) THEN
	      CALL PPLCMD (' ',' ',0,
     .		'FLOWLINE/OVER/' //ppl_wait//' '//ppl_buff, 1, 1 )

	   ELSE
	      CALL PPLCMD (' ',' ',0,
     .		'FLOWLINE/LATCH/'//ppl_wait//' '//ppl_buff, 1, 1 )
	   ENDIF
        ENDIF

* restore axes? (5/29/97  revised 3/2002
        IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* clear the overriding opacity
        opacity = -1.0
        CALL SET_OVERRIDE_OPACITY(opacity)

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* release the memory used for axis buffers
 1000 	CALL FREE_WS_DYNMEM( ax1_work )
	CALL FREE_WS_DYNMEM( ax2_work )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5400	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to 3', *5000 )
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', *1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )
 5500   CALL ERRMSG( ferr_out_of_range, status,
     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .          'Value from 0 to 100 must be given', *1000 )
	END
