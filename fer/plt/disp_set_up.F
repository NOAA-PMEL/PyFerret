	SUBROUTINE DISP_SET_UP( memory,
     .				vector, shade, use_cell, aspect, overlay, 
     .				transpz, no_labels, no_x_rng, valx1, 
     .				density, is_logx, is_logy, mr_list, nmv, 
     .				cx_list, uvar, s1_axis, s2_axis, pen, inc_grid,
     .				status )

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
* set up for contour and vector plots by loading data and title and 
* defining axis formats

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - based on CONTOUR_CMND_DATA rev 0.1
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 5/11/87 - use list of mrs - compound titles and colors
* revision 1.0 - 6/23/87 - separated initialization code into DISP_INIT
*			   incorporated SUBR. AXIS_INTERVALS and AXIS_FORMAT to
*			   make symmetric calculations for both x and y axes 
*			   in order to permit z-t plots.  Put key/label on 
*			   overlaid plots.  Added "/transposed" option.
* revision 1.1 - 7/8/87  - separated labels for normal dimensions and data set
*			   into DISP_LABELS.  general clean-up
* revision 1.2 - 7/21/87 - added calls to AXIS_ENDS to specify precise axis ends
* revision 1.3 -  8/7/87 - added XPPLUS.CMN to pass icode to LOAD_Y_VECTOR
* revision 1.4 - 3/18/88 - added grid number and limits to AXIS_FORMAT for taxis
* revision 2.0 - 4/14/88 - added do_labels argument for /NOLABELS option
*			 - loaded Y vector component from this routine
*			 - added aspect-ratio related vector rotations
* revision 2.1 - 4/28/88 - return valx1
* revision 2.2 -  9/8/88 - added relative position flag to PUT_LABEL calls
* revision 2.3 - 9/23/88 - decision to output labels now done in PUT_LABEL
*			 - time axis style determines possible "year" display
* revision 2.4 - 1/25/89 - for overlay labels /NOLABELS is acted on here (messy)
* revision 2.5 - 2/16/89 - kludge fix to improper # dimension check
* V200:  5/19/89 - 4D symmetrical
*		 - pass uvar: use exact axis limits if given as command
*		   mods or set region
*	  1/4/89 - major mods to incorporate keys on plots, pass axes to PLOT+
*	  5/7/90 - bug fix in vector length adjustment key (don't use PPL_BUFF)
* V230:  6/12/92 - trap PLOT+ nsize limitation before passing data
*                  (fixed bug in error trapping 11/27/92)
*       10/18/92 - bug fix: /NOLABELS prevented axis endpoints and styles
*                  from being set
*        1/28/93 - allow overlays on 0 and 1D promoted regions (plotify_grp_cx)
*                - and allow all values to be bad in contour, shade, etc.
* V301:	10/26/93 - added /XLIMITS and /YLIMITS axis range specifications
*	11/ 3/93 - added /TITLE
*	 1/13/94 - trap attempts to CONTOUR or SHADE 1D region as errors
*	 2/10/94 - added /PEN= to control color easily
* V320: 5/94 - array "memory" as a calling argument
* 	9/94: restrict to 99% of NSIZE buffer - kludge fix to PLOT+ bug
*      12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V401: 6/23/95 - tag the movable labels by passing ppl_movlab minus nnn
* V420: 11/95 - change to args in AXIS_FORMAT for better time axis control
*	 2/96 - /ASPECT needs to be sensitive to /XLIMITS and /YLIMITS
*		(moved data rotating and loading to a point AFTER axis set up)
*		/ASPECT needs to be similarly sensitive to /OVERLAY 
*		==> COMMON XWINDOW_STATE has been created **temporarily**
*		    with the intention that it will be expanded in future
*		    so overlays can be placed on **previously** drawn windows
* V430: 5/96 - fixed up error message text
* Linux Port *kob* 1/97 - include new cmn file xwindow_state.cmn because
*			  locally declared common blocks aren't allowed.
* V450: 3/97 - support for curvilinear coordinate plots
*       8/97 - added curvilinear check to make sure that args 2 and 3 lie in
*              compatible planes with the variable to be plotted
* V500  4/99 - *jd* Ensure PPL knows plot is overlay: set SOVER flag here
*       6/99 - save time axis info in XPLOT_STATE to enable line plot overlays
*		(with plot/vs and polygon)
* V510	12/99 - *sh* plot of all bad data no longer an error
*        3/00 - *sh* /NOLABELS to avoid xlab & ylab
*	 5/00 - added /HLIMITS and /VLIMITS (denigrating /xlimits/ylimits)
* V520   6/00 *sh* - bug fix: trap ndim>2 on overlays
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
* V530 *acm* 1/01  cal_id for alternative calendar axes.
* V533 *acm* 6/01 for LOG plots add logicals is_logx, is_logy.  set to plot
*                   log of depth as inverse axis
*       ACM  8/2001 BUG FIX: save cal id if vertical time axis
*      *acm* 10/01  remove VMS includes
*      *acm* 12/01  Allocate PPLUS memory for the flag array used by
*                   flowline.
* V541 *acm*  4/02  for flowlines, more PPLUS memory was allocated than needed.
*      *acm*  4/02  If this is an overlay, and there is no valid data, add
*                   this information variable name in the key below the plot.
* v541 *acm*  8/02  For "No Valid Data" the test is: 
*                    IF (no_x_rng .AND. (valx1.EQ.bad_val4) )
* v5.5 *acm* 11/02  Set PPLUS memory for the cells array used by cellarray 
*                   in SHADE plots
*      *acm* 12/02  When setting has_time_axis flag, only do it if an
*                   axis being used IS a time axis
* v5.51*acm*  1/03  Comment out the cellarray implementation; xgks doesnt
*                      completely implement it & has a precision bug for 
*                      large plots.  see comment lines: c no_cellarray
* V552 *acm*  3/03  Send the key string to key_loc since it may be multi-line
*      *acm*  5/03  Send the flag irregular to AXIS_INTERVALS.  
* v5.52*acm*  6/03  Restore cellarray setup. uses its_gif flag so these are
*                   only done in -gif mode
* v5.53*acm*  6/03  send correct axis to AXIS_INTERVALS
* V580 *acm* 12/04 Fix bug 1129; formatting axis for small lon/lat ranges:
*                  send delta to routine AXIS_FORMAT
* V600 *acm* 10/05 Fix bug 1347; for very fine axes, set the flag irregular in
*                  AXIS_INTERVALS, and compute the delta coordinate. Then after
*                  the call to pplldc, reset the plot type to regular if the
*                  grid is in fact regular.
* V608 *acm* 09/07 Define symbols AX_HORIZ and AX_VERT with the x,y,z,t direction 
*                  of the axes as they will appear on the plot.
* V62  *acm* 03/09 Fix bug 1649: get the value of cal_id for this time axis
*                  BEFORE using it in evaluating arguments to /hlim or /vlim 
* V630 *acm*  9/09 Introduction of syntax for variance-based and histogram levels
* V663 *acm* 07/10 If openlevels computation fails (very-large value data fields)
*                  then issue a warning and do linear levels instead.
* v68  *acm* 4/08 Allow shade, vector plots of degenerate regions - world region 
*              within a grid cell
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* V685  *acm* 9/13 Fixing ticket 2092. Check that the pplus internal storage
*                  for coordinate data will be enough for the grid and if not
*                  return an error message. 
* V686 *acm* 12/13 Symbols AX_HORIZ and AX_VERT may be x,y,z,t,e,f  
* V691  *acm*  5/14 Clean up temp memory used for computing histogram levels
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V695+ *sh*  3/15 if an FMRC dataset disambiguate T and F axes with labels
* V695  *acm* 9/15 send REAL*4 rbad to compare with lev_max, lev_min inside std
* V697  *acm* 2/16 Fix ticket 2360: overlays on time plots
* V697  *acm* 2/16 Fix ticket 2365: overlays with a single vector (or single-valued variable)
*                  returned a status flag of 0 from COMPUTE_MNSTD. Set status to ferr_ok
*                  after handling the condition.
* V701  *acm* 8/16 Fix ticket 2463: single-valued data with a /LEVELS setting
* V701  *acm* 8/16 Fix ticket 2468: Symbols AX_HORIZ and AX_VERT are not reset on an /OVERLAY plot
* V702  *acm* 10/16 Test the results of FIELD_MINMAX routine before use; if all data is 
*                   all-missing it returns arbitrary_large_val4.

C replace includes with those from plot_set_up, 
C plus those starting w/ xbox_bounds:

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_setup.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xtext_info.cmn'
        include 'axis_inc.decl'    ! with axis lengths
        include 'AXIS.INC'         ! with axis lengths
	include 'PPLDAT.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'

	include 'xbox_bounds.cmn'
	include 'xcurvilinear.cmn'
	include 'command.parm'
	include 'xcommand.cmn'
	include 'xpplus.cmn'


*kob 1/97
	include 'xwindow_state.cmn'


* calling argument declarations:
	LOGICAL		vector, overlay, transpz, no_labels, no_x_rng,
     .			is_logx, is_logy, flows, shade, inc_grid
	INTEGER		nmv, mr_list( nmv ), cx_list( nmv ), uvar,
     .			density, pen, status
	REAL		memory( mem_blk_size, max_mem_blks )
	REAL 		aspect, valx1, s1_axis(*), s2_axis(*)

* internal variable declarations:
	CHARACTER*1020	COMPOUND_TITLE, COMPOUND_KEY, plot_title
	CHARACTER	TM_FMT*16, LEFINT*8, LEFT_REAL*16, INTERNAL_WHOI_DATE*14, 
     .                  xstyle*3, ystyle*3, cmnd_name*8, year*4, buff*8,
     .			letter*1, buff1*16, buff2*16, buff30*30,
     .                  TM_GET_CALENDAR_NAME*32, cal_name_1*32, 
     .                  cal_name_2*32, err_string*128,
     .			xtra_xlab*24, xtra_ylab*24, t1_date*14

        CHARACTER*1     axdir(6)

	LOGICAL		NO_GRID_RANGE, GEOG_LABEL, TM_FPEQ, ITS_FMRC,
     .			user_title, time_axis, time_axis_dim1, timeax_both, 
     .			regrid_1, regrid_2, no_y_rng, edges_ok,
     .			ITS_GIF_GRAPHICS, its_gif, REGULARLY_SPACED,
     .                  irregular
        LOGICAL         use_cell

	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_SIZE,
     .			dim1, dim2, mvx, mvx_temp, mvy, mvy_temp,
     .			ctx, cty, xblk1, yblk1, plot_mem_used, nvars,
     .			lo1, lo2, hi1, hi2, size1, size2, size,
     .			n1_regrid, n2_regrid, grid, slen, pxlim, pylim,
     .			style, justify, ipen, ndim, dims(nferdims), ptitl,
     .			arg_xpos, cx_xpos, cx_ypos, mr_xpos, mr_ypos,
     .			phlim, pvlim, slen2, cal_id, tax,
     .			x_axtyp, y_axtyp, iscl, siz, nsiz, 
     .			TM_GET_CALENDAR_ID, mode_num, axis1, axis2,
     .                  hblk1, mvh_temp
	REAL		s, w1_lo, w1_hi, w2_lo, w2_hi, valy1,
     .			d1_regrid, d2_regrid, bad_x, bad_y, rlo, rhi,
     .			xkeypos, ykeypos, xkeysize, ykeysize, lab_ht,
     .                  first, last
	REAL*8		xaxlo, xaxhi, yaxlo, yaxhi, delta
	REAL*4		rw1lo, rw2lo, rbad, dt_min

        DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			'setting up 2D plot', 18 )

* initialize
	mvx	  = mr_list( 1 )
	ctx       = cx_list  ( 1 )
	grid	  = mr_grid( mvx )
	bad_x	  = mr_bad_data( mvx )
	cal_id    = 1
	x_axtyp   = 1
	y_axtyp   = 1
        plot_mem_used = 0
	mvx_temp  = 0

	IF ( vector ) THEN
	   mvy	  = mr_list( 2 )
	   cty	  = cx_list  ( 2 )
	   bad_y  = mr_bad_data( mvy )
	ENDIF
	phlim = qual_given( slash_hlimits )
	pvlim = qual_given( slash_vlimits )
	IF ( transpz ) THEN
	  pxlim = qual_given( slash_ylimits )
	  pylim = qual_given( slash_xlimits )
	ELSE
	  pxlim = qual_given( slash_xlimits )
	  pylim = qual_given( slash_ylimits )
	ENDIF
	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0

* get 4D shape of data region to load (already oversized)
	dim1 = plot_axis(1)
	dim2 = plot_axis(2)
	axis1 = grid_line(dim1, grid)
	axis2 = grid_line(dim2, grid)
	size1 = CX_DIM_LEN( dim1, ctx )
	size2 = CX_DIM_LEN( dim2, ctx )
	size  = size1 * size2


* determine PLOT+ memory required 

        plot_mem_used = CGRID_SIZE( cx_list(1) ) + size1 + size2
        IF ( vector ) 
     .        plot_mem_used = plot_mem_used * 2

	flows     = qual_given( slash_vect_flowline  ).GT.0

!  For flowlines, more PLOT+ memory needed; based on density parameter and
!  grid size.

        IF ( flows ) THEN
           nsiz = MAX(size1, size2)
           iscl = 200/nsiz
           iscl = iscl + density - 5
           IF (iscl .LT. 1) iscl = 1
           IF (iscl .GT. 10) iscl = 10

C   set up memory for the flag grid

           siz = (nsiz*nsiz*(iscl*iscl +2)) + nsiz*iscl   ! see flowline
           plot_mem_used = plot_mem_used + siz
        ENDIF

* set flag indicating a 2D plot is on the screen
	twodee_on = .TRUE.

* save axis pointers so PLOT+ can get box boundaries
	bb_grid = grid

* On a plot/set, the intervening commands may change the default grid.
* Increment the grid. It will be decremented at the end of xeq_pplus

	bb_grid_inccount = 0
	IF (inc_grid) THEN
	   bb_grid_inccount = 1
	   CALL TM_USE_DYN_GRID(grid)  ! see ticket 883
	ENDIF

	IF ( transpz ) THEN
	   bb_yaxis = dim1
	   bb_xaxis = dim2
	ELSE
	   bb_xaxis = dim1
	   bb_yaxis = dim2
	ENDIF

* check for valid data plane
	CALL GET_CX_DIMS( ctx, ndim, dims )
	IF ( ndim .GT. 2 ) GOTO 5160
	IF ( overlay ) THEN
* *sh* 1/93 - restrictions less: coords merely have to be valid in 2D plane
           IF ( cx_lo_ss(ctx,dim1) .EQ. unspecified_int4
     .     .OR. cx_lo_ww(dim1,ctx) .GE. cx_hi_ww(dim1,ctx) ) GOTO 5150
           IF ( cx_lo_ss(ctx,dim2) .EQ. unspecified_int4
     .     .OR. cx_lo_ww(dim2,ctx) .GE. cx_hi_ww(dim2,ctx) ) GOTO 5150
	ELSEIF ( vector) THEN     ! 1/94
	   IF ( nplot_axis .GT. 2 ) GOTO 5160
	ENDIF

* check # of variables and consistency in X and Y component dimensions
	IF ( vector ) THEN
	   nvars = 2
	   IF ( num_uvars_in_cmnd .EQ. 4 ) THEN
	      curvilinear = .TRUE.
	      arg_xpos = 3		! 3rd cmd arg is X positions
	   ELSE
	      curvilinear = .FALSE.
	      IF ( num_uvars_in_cmnd .GT. 2 ) GOTO 5300
	   ENDIF
	   IF ( CX_DIM_LEN(dim1,ctx) .NE. CX_DIM_LEN(dim1,cty)
     .	  .OR.  CX_DIM_LEN(dim2,ctx) .NE. CX_DIM_LEN(dim2,cty) )
     .		CALL ERRMSG( ferr_inconsist_plane, status,
     .			'X and Y vector components', *5100 )
	ELSE
	   nvars = 1
	   IF ( num_uvars_in_cmnd .EQ. 3 ) THEN
	      curvilinear = .TRUE.
	      arg_xpos = 2
	   ELSE
	      curvilinear = .FALSE.
	      IF ( num_uvars_in_cmnd .GT. 1 ) GOTO 5300
	   ENDIF
	ENDIF

* If its a multi-argument curvilinear request check that the xpos, ypos
* fields are conformable with the field of values
	IF ( curvilinear ) THEN
	   IF (.NOT. overlay) use_keys = .FALSE.
	   edges_ok = cmnd_num.EQ.cmnd_shade
	   mr_xpos = mr_list(arg_xpos)
	   mr_ypos = mr_list(arg_xpos + 1)
	   cx_xpos = cx_list(arg_xpos)
	   cx_ypos = cx_list(arg_xpos + 1)
* Note that the plot axes have never been transposed by this point
	   IF ( CX_DIM_LEN(plot_axis(3),cx_xpos) .GT. 1
     .	   .OR. CX_DIM_LEN(plot_axis(4),cx_xpos) .GT. 1
     .	   .OR. CX_DIM_LEN(plot_axis(1),cx_xpos) .EQ. 1 ) THEN
	      letter = 'X'
	      GOTO 5400
	   ELSEIF ( CX_DIM_LEN(plot_axis(3),cx_ypos) .GT. 1
     .	   .OR. CX_DIM_LEN(plot_axis(4),cx_ypos) .GT. 1
     .	   .OR. CX_DIM_LEN(plot_axis(2),cx_ypos) .EQ. 1 ) THEN
	      letter = 'Y'
	      GOTO 5400
	   ENDIF
	ENDIF

* create temporary buffers to contain the plot data
	CALL CREATE_TEMP_MEM_VAR( ctx, mvx_temp, status )
	IF ( status .NE. ferr_ok ) RETURN
	xblk1 = mr_blk1( mvx_temp )
	IF ( vector ) THEN
	   CALL CREATE_TEMP_MEM_VAR( cty, mvy_temp, status )
	   IF ( status .NE. ferr_ok ) GOTO 5010
	   yblk1 = mr_blk1( mvy_temp )
	ENDIF

* copy the plot data to the buffer(s) - possibly transposing
	IF ( transpz ) THEN
	   CALL COPY_GRID_TRANS( size1, size2,
     .				 memory(1, mr_blk1(mvx)), mvx,
     .				 memory(1, xblk1       ), mvx_temp )
	   IF ( vector )
     .		CALL COPY_GRID_TRANS(   size1, size2,
     .					memory(1, mr_blk1(mvy)), mvy,
     .					memory(1, yblk1       ), mvy_temp )
	ELSE
	   CALL COPY_GRID( memory(1, mr_blk1(mvx)), mvx,
     .			   memory(1, xblk1       ), mvx_temp )
	   IF ( vector )
     .		CALL COPY_GRID(	memory(1, mr_blk1(mvy)), mvy,
     .				memory(1, yblk1       ), mvy_temp )
	ENDIF
	
* determine plotting limits
	IF ( transpz ) THEN
	   dim2 = plot_axis(1)
	   dim1 = plot_axis(2)
	   axis1 = grid_line(dim1, grid)
	   axis2 = grid_line(dim2, grid)
	ENDIF
	lo1  = cx_lo_ss( ctx, dim1 )
	hi1  = cx_hi_ss( ctx, dim1 )
	lo2  = cx_lo_ss( ctx, dim2 )
	hi2  = cx_hi_ss( ctx, dim2 )
	size1 = hi1 - lo1 + 1
	size2 = hi2 - lo2 + 1

* define special axis labels to disambiguate T and F axes
	xtra_xlab = ' '
	xtra_ylab = ' '
        IF ( ITS_FMRC(grid) ) THEN
* ... add these labels only if the plot has two calendar axes
	   IF ( GEOG_LABEL(t_dim,grid)
     .	 .AND.  GEOG_LABEL(f_dim,grid)
     .	 .AND.  (dim1.EQ.t_dim .OR. dim1.EQ.f_dim)
     .	 .AND.  (dim2.EQ.t_dim .OR. dim2.EQ.f_dim) ) THEN
	      IF (dim1 .EQ. t_dim) THEN
	         xtra_xlab = 'Verification Time ==>'
	         xtra_ylab = 'Initialization Time ==>'
	      ELSE
	         xtra_xlab = 'Initialization Time ==>'
	         xtra_ylab = 'Verification Time ==>'
	      ENDIF
	   ENDIF
	ENDIF

* Define symbols AX_HORIZ and AX_VERT with the x,y,z,t direction 
* of the axes as they will appear on the plot.

	IF (.NOT.overlay) THEN
        buff1 = axdir(dim1)
	CALL PPLCMD ( from, line, 0, 'SET AX_HORIZ '//buff1, 1, 1 )

        buff1 = axdir(dim2)
        CALL PPLCMD ( from, line, 0, 'SET AX_VERT '//buff1, 1, 1 )
	ENDIF

* check that there is some valid data
	no_x_rng = NO_GRID_RANGE( memory(1,xblk1), size, mvx_temp, valx1 )
	IF ( no_x_rng .AND. valx1 .EQ. bad_x ) valx1 = bad_val4
	IF ( vector ) THEN
	   no_y_rng = NO_GRID_RANGE( memory(1,yblk1), size, mvy_temp, valy1 )
	   IF ( no_y_rng .AND. valy1 .EQ. bad_y  ) THEN
	      valx1 = bad_val4
	      no_x_rng = no_y_rng	! need both to have either
	   ENDIF
	ENDIF

* determine axis re-gridding needs

        irregular = .NOT. ( line_regular(axis1) )
	CALL AXIS_INTERVALS( 'X', mvx_temp, ctx, dim1, curvilinear, s1_axis,
     .			     mr_lo_ss(mvx_temp,dim1), mr_hi_ss(mvx_temp,dim1),
     .			     w1_lo, w1_hi, n1_regrid, d1_regrid, regrid_1,
     .                       irregular )

        irregular = .NOT. ( line_regular(axis2) )
	CALL AXIS_INTERVALS( 'Y', mvx_temp, ctx, dim2, curvilinear, s2_axis,
     .			     mr_lo_ss(mvx_temp,dim2), mr_hi_ss(mvx_temp,dim2),
     .			     w2_lo, w2_hi, n2_regrid, d2_regrid, regrid_2,
     .                       irregular )

	IF ( regrid_1 .OR. regrid_2 ) THEN
	   icode = 1
	ELSE
	   icode = 0
	ENDIF

!  For SHADE plots, more PLOT+ memory needed for the cells array
!  used in cellarray call.  (s1_axis and s2_axis have been set by now.)

        its_gif = ITS_GIF_GRAPHICS()

        irregular = .NOT. ( line_regular(axis1) )
        IF (.NOT. irregular)
     .      irregular = .NOT. ( line_regular(axis2) )

        use_cell  = .FALSE.

        IF ( shade  .AND. 
     .       (.NOT.overlay) .AND. 
     .       (its_gif) .AND. 
     .       (.NOT.mode_metafile) .AND.
     .       (.NOT.curvilinear) .AND. 
     .       (.NOT.irregular) ) use_cell = .TRUE.
        IF (use_cell) THEN
           nsiz = size1* size2
           plot_mem_used = plot_mem_used + nsiz
        ENDIF

* For 3-argument curvilinear plots, see if we want to 	 
* do modulo replications of the data.	 
 	 
         IF (curvilinear) xmodulo = qual_given(slash_shad_mod) .GT. 0	 
     .                        .OR. qual_given(slash_cont_mod) .GT. 0   
     .                        .OR. qual_given(slash_polygon_mod) .GT. 0  
     .                        .OR. qual_given(slash_vect_mod) .GT. 0  

* set up PLOT+ to ignore bad data flag
	ppl_buff = ' '
	WRITE ( ppl_buff, 3005 ) mr_bad_data( mvx )
 3005	FORMAT ( 'LIMITS ',G15.8,' ZEQ' )
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

* assign a color for the data plotting
!	IF ( vector ) THEN
!	   ipen = nvect_on+1
!	ELSE
!	   ipen = ncont_on+1
!	ENDIF
	IF ( pen .LT. 0 ) THEN    ! 2/94
	   ipen = nvect_on + ncont_on + 1
	ELSE
	   ipen = pen
	ENDIF
	WRITE ( ppl_buff, 3105) ipen
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3105	FORMAT('PEN 1,',I3)

* set up the axis formats (use exact limits, if given by /HLIMITS or /VLIMITS)

	time_axis = GEOG_LABEL( dim1, grid )  .AND. 
     .                 (dim1.EQ.t_dim .OR. dim1.EQ.f_dim) 

	IF ( .NOT.overlay ) has_time_axis = time_axis	! save for PLOT/VS/OVERLAY (6/99)
	IF (time_axis) THEN
           tax = grid_line(dim1, grid)
           cal_id = TM_GET_CALENDAR_ID ( line_cal_name(tax) )
           IF(overlay .AND. cal_id .NE. saved_calendar_id) GO TO 5660
           
	   IF ( .NOT.overlay ) THEN
	      saved_calendar_id = cal_id
	      ppl_buff = ' '
              WRITE ( ppl_buff, 3004 ) line_cal_name(tax)
              CALL PPLCMD (from, line, 0, ppl_buff, 1, 1)
 3004         FORMAT ('CALENDAR ', A7)
	   ENDIF

* Send the start time and time units for overlay to PPLUS
	   IF (overlay) THEN
	      t1_date =  INTERNAL_WHOI_DATE(grid, dim1, 1.0D0 )
	      dt_min = line_tunit( grid_line(dim1,grid) ) / 60.
	      CALL SET_T1_DATE (t1_date, dt_min)
	   ENDIF
        ELSE
	
* Time on the vertical axis?
	   IF (GEOG_LABEL( dim2, grid )  .AND. 
     .                  (dim2.EQ.t_dim .OR. dim2.EQ.f_dim)) THEN

* Send the start time and time units for overlay to PPLUS
	      IF (overlay) THEN
	         t1_date =  INTERNAL_WHOI_DATE(grid, dim2, 1.0D0 )
	         dt_min = line_tunit( grid_line(dim2,grid) ) / 60.
	         CALL SET_T1_DATE (t1_date, dt_min)
	      ENDIF
	   ENDIF

	ENDIF


	IF ( .NOT.overlay ) THEN

	   IF ( pxlim .GT. 0 ) THEN
	      IF (.NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
              tax = grid_line(f_dim, grid)
              IF (tax .EQ. 0) tax = grid_line(f_dim, grid)
              cal_id = TM_GET_CALENDAR_ID ( line_cal_name(tax) )
	      CALL EQUAL_RANGE_T( cmnd_buff(qual_start(pxlim):qual_end(pxlim)),
     .				dim1, grid, xaxlo, xaxhi, delta,
     .				cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( phlim.GT.0 ) THEN
	      CALL EQUAL_RANGE_T(cmnd_buff(qual_start(phlim):qual_end(phlim)),
     .				 bb_xaxis, grid, xaxlo, xaxhi, delta,
     .				 cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( curvilinear ) THEN
	      CALL FIELD_MINMAX( memory(1,mr_blk1(mr_xpos) ), 
     .				mr_xpos, cx_xpos, rlo, rhi )

	      IF (rlo .EQ. arbitrary_large_val4) THEN ! all missing
	         xaxlo = -1
	         xaxhi = 1
	      ElSE
	         xaxlo = rlo
	         xaxhi = rhi
	      ENDIF
	      IF (xaxlo .EQ. xaxhi) THEN
		 letter = 'X'
		 GOTO 5500
	      ENDIF
	      delta = unspecified_val8
	   ELSE
	      CALL CHOOSE_CX_WW( uvar, cx_cmnd, ctx, dim1, xaxlo, xaxhi, size1 )
              IF (shade_trim) CALL TRIM_AXIS_ENDS( uvar, cx_cmnd, ctx, 
     .                        dim1, xaxlo, xaxhi )
              IF (xaxlo .GE. xaxhi) GOTO 5160
	      delta = unspecified_val8
	   ENDIF

	   haxlo = xaxlo	! save limits so overlays
	   haxhi = xaxhi	! will know them

	   CALL AXIS_ENDS  ( 'X', dim1, grid, xaxlo, xaxhi, delta, is_logx,
     .			      x_axtyp, status )
	   IF ( status .NE. ferr_ok ) THEN
              first = xaxlo
              last  = xaxhi
              GOTO 5700
           ENDIF

	   timeax_both = .FALSE.
	   CALL AXIS_FORMAT( 'X', dim1, grid, xaxlo, xaxhi, delta,
     .                       curvilinear, cx_xpos,
     .                       time_axis,timeax_both,xstyle,
     .			     no_labels, xtra_xlab)

           time_axis_dim1 = time_axis

* second axis

	   time_axis = GEOG_LABEL( dim2, grid )  .AND. 
     .                  (dim2.EQ.t_dim .OR. dim2.EQ.f_dim)

	   has_time_axis = has_time_axis .OR. time_axis ! 6/99

C ACM added 8/2001 save cal id if vertical time axis
           IF (time_axis) THEN
              tax = grid_line(dim2, grid)
              cal_id = TM_GET_CALENDAR_ID ( line_cal_name(tax) )
	      IF(overlay .AND. cal_id .NE. saved_calendar_id) GO TO 5660
	      ppl_buff = ' '
	      WRITE ( ppl_buff, 3004 ) line_cal_name(tax)
              CALL PPLCMD (from, line, 0, ppl_buff, 1, 1)
	      saved_calendar_id = cal_id
           ENDIF

	   IF ( pylim .GT. 0 ) THEN
	      IF (.NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
	      CALL EQUAL_RANGE_T( cmnd_buff(qual_start(pylim):qual_end(pylim)),
     .				dim2, grid, yaxlo, yaxhi, delta,
     .				cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( pvlim.GT.0 ) THEN 
	      CALL EQUAL_RANGE_T(cmnd_buff(qual_start(pvlim):qual_end(pvlim)),
     .				 bb_yaxis, grid, yaxlo, yaxhi, delta,
     .				 cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( curvilinear ) THEN
	      CALL FIELD_MINMAX( memory(1,mr_blk1(mr_ypos) ),
     .				mr_ypos, cx_ypos, rlo, rhi )
	      IF (rlo .EQ. arbitrary_large_val4) THEN ! all missing
	         yaxlo = -1
	         yaxhi = 1
	      ElSE
	         yaxlo = rlo
	         yaxhi = rhi
	      ENDIF

	      IF (yaxlo .EQ. yaxhi) THEN
		 letter = 'Y'
		 GOTO 5500
	      ENDIF
	      delta = unspecified_val8
	   ELSE
	      CALL CHOOSE_CX_WW( uvar, cx_cmnd, ctx, dim2, yaxlo, yaxhi, size2 )
              IF (shade_trim) CALL TRIM_AXIS_ENDS( uvar, cx_cmnd, ctx, 
     .                        dim2, yaxlo, yaxhi )
              IF (yaxlo .GE. yaxhi) GOTO 5160
	      delta = unspecified_val8
	   ENDIF

	   vaxlo = yaxlo	! save limits so overlays
	   vaxhi = yaxhi	! will know them

	   CALL AXIS_ENDS  ( 'Y', dim2, grid, yaxlo, yaxhi, delta, is_logy,
     .			      y_axtyp, status )
	   IF ( status .NE. ferr_ok ) THEN
              first = xaxlo
              last  = xaxhi
              GOTO 5700
           ENDIF 
	   timeax_both = (time_axis .AND. time_axis_dim1) 
	   CALL AXIS_FORMAT( 'Y', dim2, grid, yaxlo, yaxhi, delta,
     .                       curvilinear, cx_ypos,
     .                       time_axis,timeax_both,ystyle,
     .			     no_labels, xtra_ylab)

*   When not an overlay, set the axis type to log or reverse log
	   IF (is_logx .OR. is_logy) THEN
	      WRITE (buff2, 3006) x_axtyp, y_axtyp

 3006         FORMAT ('axtype,', I2, ',', I2)
	      CALL PPLCMD ( from, line, 0, buff2, 1, 1)
	   ENDIF

        ENDIF   ! .NOT.overlay

* rotate vectors to make axis scale ratio correction ?
	IF ( vector .AND. aspect.GT.0. ) CALL RE_ORIENT(
     .			haxlo, haxhi, vaxlo, vaxhi, dim1, dim2,
     .			size1, size2, transpz, aspect,
     .			memory(1, xblk1), mvx_temp,
     .			memory(1, yblk1), mvy_temp, s )
	if (overlay) SOVER = .true.


* The coordinates are loaded into PPLUS arrays. Check that the grid
* isn't too large.
	CALL CHECK_2D_SIZE (size1, size2, err_string, status)
	IF (status .NE. ferr_ok) GOTO 5720

* load contour data into PPLUS and (possibly) regrid to regular grid
	rw1lo = w1_lo
	rw2lo = w2_lo

	CALL PPLLDC_envelope ( icode, memory(1, xblk1), size1, size2,
     .		      1, size1, 1, size2, s1_axis, s2_axis, 
     .		      n1_regrid, n2_regrid, rw1lo, rw2lo,
     .		      d1_regrid, d2_regrid, plot_mem_used)

* load curvilinear pointers and sizes for later access from PLOT+
	IF ( curvilinear ) THEN
	   wwxmin = xaxlo
	   wwxmax = xaxhi
	   wwymin = yaxlo
	   wwymax = yaxhi
	   CALL LOAD_CURVILINEAR( memory(1,mr_blk1(mr_xpos)),
     .				  memory(1,mr_blk1(mr_ypos)),
     .				  dim1, dim2, arg_xpos,
     .				  edges_ok, mr_xpos, mr_ypos,
     .				  cx_xpos, cx_ypos, ctx, status)
	   IF ( status .NE. ferr_ok) GOTO 5100
	ENDIF

* Compute data mean and if needed for the levels setting the std, store
* in common.  If the user set a std to use, store that.
* send 	REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* If the user set some /LEVELS then compute_mnstd just returns 

       nsiz = size1* size2
       rbad = bad_val4
       CALL COMPUTE_MNSTD (memory(1, xblk1), bad_x, need_std, 
     .       nsiz, rbad, status)
       IF (need_std .AND. status.NE.ferr_ok) THEN
* ... set up for automatic levels
          CALL PPLCMD ( from, line, 0, 'LEV,()', 1, 1 )
          CALL USE_LINEAR_LEVELS
          status = ferr_ok
       ENDIF

* If histogram-based levels are requested, compute the 
* histogram bins.

       IF (need_histo) THEN
* create temporary buffer to contain workspace
	  CALL CREATE_TEMP_MEM_VAR( ctx, mvh_temp, status )
	  IF ( status .NE. ferr_ok ) RETURN
	  hblk1 = mr_blk1( mvh_temp )
          plot_mem_used = plot_mem_used + nsiz

	  CALL compute_histo_bins (memory(1,xblk1), memory(1,hblk1), 
     .       bad_x, nsiz, status) 
     
* ... clean up temporary variable
	  CALL DELETE_VARIABLE( mvh_temp )
       ENDIF

* For cases where regrid was set in AXIS_INTERVALS not because of a 
* subspan modulo axis, but because the axis precsion is too small for 
* single precision representation, now we reset the plot type to regular 
* if the axes are in fact regular.

        IF (line_regular(axis1) .AND. line_regular(axis2)) 
     .      CALL SET_REGULAR_PLOT

	CALL DELETE_VARIABLE( mvx_temp )

	IF ( vector ) THEN
* load the Y vector component
	   CALL PPLLDV_envelope ( icode, memory(1,yblk1), size1, size2,
     .		      1, size1, 1, size2)
	   CALL DELETE_VARIABLE( mvy_temp )
	ENDIF
	
	IF ( no_labels ) RETURN

! ====================== LABELS ========================

* if we are starting from a blank screen compose a title
* if we are overlaying on a plot put the new variable name on as a key
	IF ( user_title ) THEN
	   CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   slen = TM_LENSTR1( plot_title )
	ENDIF
	IF ( overlay ) THEN
* ... cook up a key label for this variable only (or pair for VECTOR)
*     If no valid data (no_x_rng) add this info to the label.
	   use_keys = .TRUE.
	   IF ( .NOT.user_title ) THEN
	      CALL OV_GRP_CX_DOC( cx_list, nmv, cx_xtra_doc )
	      plot_title = COMPOUND_KEY( cx_list, nvars, cx_xtra_doc, slen )
	   ENDIF
           IF (no_x_rng .AND. (valx1.EQ.bad_val4) ) THEN
              plot_title(slen+1:slen+14) = ' No Valid Data'
              slen = slen + 14
           ENDIF
	   CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, justify, plot_title(:slen) )
	   CALL PUT_CMND_KEY( cmnd_num, plot_title(:slen),
     .			      xkeypos, ykeypos, xkeysize, ykeysize,
     .			      lab_ht )
	ELSE
	   IF ( .NOT. user_title ) THEN
	      IF ( use_keys ) THEN
	         plot_title = COMPOUND_KEY( cx_list, nvars, key_doc, slen )
	      ELSE
	         plot_title = COMPOUND_TITLE( cx_list, nvars, slen )
	      ENDIF
	   ENDIF
	   CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*xlen, letsize_title,
     .			   letsize_title, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
* ... see if a "year" label is required
	   year = unspecified_name4
	   IF (dim1.EQ.t_dim.AND.(xstyle.EQ.'MON'.OR.xstyle.EQ.'DAY')) THEN
	      CALL TSTEP_TO_DATE( grid, t_dim, xaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ELSEIF (dim1.EQ.f_dim.AND.(xstyle.EQ.'MON'.OR.xstyle.EQ.'DAY')) THEN
	      CALL TSTEP_TO_DATE( grid, f_dim, xaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ELSEIF (dim2.EQ.t_dim.AND.
     .                  (ystyle.EQ.'MON'.OR.ystyle.EQ.'DAY')) THEN
	      CALL TSTEP_TO_DATE( grid, t_dim, yaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ELSEIF (dim2.EQ.f_dim.AND.
     .                  (ystyle.EQ.'MON'.OR.ystyle.EQ.'DAY')) THEN
	      CALL TSTEP_TO_DATE( grid, f_dim, yaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ENDIF
* ... put on the basic position and data set labels

* *ACM* if time axis, then cal_id is set to the calendar.  Label the
*       plot with the calendar name if there is non-standard calendar

	   CALL DISP_LABELS( cx_plot, year, cal_id )
	ENDIF

* special remarks for vector plots
* ... "scale ratio adjustment: nnn"
* note: "plot_title" is just used as a buffer
	IF ( vector ) THEN
	   IF ( aspect .GT. 0.0 ) THEN
	      plot_title = 'adjustment for differing X/Y scales: '//
     .			  TM_FMT(s,3,16,slen)
	      CALL PUT_LABEL( ppl_movlab-ppl_aspect, 0.0, +0.2,
     .			      ppl_left_just, 0.09, 
     .			      plot_title(:37+slen),
     .			      lab_loc_absolute, lab_loc_axlen_plus )
	   ENDIF
* ... offset components (staggered vectors)?
	   IF (   cx_lo_ww(dim1,ctx) .NE. cx_lo_ww(dim1,cty)
     .	     .OR. cx_hi_ww(dim1,ctx) .NE. cx_hi_ww(dim1,cty)
     .	     .OR. cx_lo_ww(dim2,ctx) .NE. cx_lo_ww(dim2,cty)
     .	     .OR. cx_hi_ww(dim2,ctx) .NE. cx_hi_ww(dim2,cty))THEN
	         CALL PUT_LABEL( ppl_movlab-ppl_offset, 8.0, +0.2,
     .			      ppl_right_just, 0.09, 
     .				'X and Y components are offset',
     .			      lab_loc_absolute, lab_loc_axlen_plus )
	   ENDIF
	ENDIF

	status = ferr_ok
	RETURN

* error exit
 5000	IF ( vector ) CALL DELETE_VARIABLE( mvy_temp )
 5010	CALL DELETE_VARIABLE( mvx_temp )
 5100	RETURN
 5150	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'overlay not valid in plane of plot'//pCR//
     .					cmnd_buff(:len_cmnd), *5100 )
 5160	IF (mvx_temp .NE. 0) CALL DELETE_VARIABLE( mvx_temp )
        CALL ERRMSG( ferr_dim_underspec, status, 
     .			'must be a 2D region'//pCR//
     .					cmnd_buff(:len_cmnd), *5100 )

! 5200	CALL ERRMSG( ferr_no_range, status,
!     .				'data are all flagged as bad', *5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		     'can only '//commands(cmnd_num)//' single fields'//pCR//
     .		     'Use /OVERLAY for multiple variables', *5100 )
 5400	CALL ERRMSG( ferr_inconsist_plane, status,
     .		 letter//' coord field lies'//pCR//
     .		'in different plane from data to be plotted', *5100 )
 5500	CALL ERRMSG( ferr_no_range, status, letter//
     .				' coordinate field', *5000 )

 5660	cal_name_1 = TM_GET_CALENDAR_NAME(saved_calendar_id)
        cal_name_2 = TM_GET_CALENDAR_NAME(cal_id)
        slen = TM_LENSTR1 (cal_name_1)
        slen2 = TM_LENSTR1(cal_name_2)
	CALL ERRMSG( ferr_inconsist_grid, status,
     .          'Differing calendar axes: '//pCR//
     .		'first variable is on '//cal_name_1(:slen)//
     .          ' axis'//pCR//
     .		'subsequent variable is on '//cal_name_2(:slen2)//
     .		' axis', *5000 )

 5700	buff1 = LEFT_REAL (first, '(G15.3)', slen)
	buff2 = LEFT_REAL (last,  '(G15.3)', slen2)
	CALL ERRMSG( ferr_out_of_range, status,
     . 'Limits for log axis negative or too small: '//
     .  buff1(:slen)// ' : '// buff2(:slen2), *5000 )

 5710   CALL ERRMSG( ferr_out_of_range, status,
     .     'data too large to compute Std Dev'//pCR//
     .      cmnd_buff(:len_cmnd), *5100 )
 5720   CALL ERRMSG( ferr_out_of_range, status,
     .     err_string//pCR//cmnd_buff(:len_cmnd), *5100 )

	END
