	SUBROUTINE DISP_SET_UP( memory,
     .				vector, aspect, overlay, transpz, no_labels,
     .				no_x_rng, valx1, mr_list, nmv, cx_list, uvar,
     .				s1_axis, s2_axis, pen, status )

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

#ifdef unix
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'command.parm'
	include 'xcommand.cmn'
	include 'xplot_setup.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xpplus.cmn'
	include 'xbox_bounds.cmn'
	include 'xcurvilinear.cmn'
	include 'axis_inc.decl'	! axis lengths
	include 'AXIS.INC'	! axis lengths
        include 'ppldat_inc.decl'       ! with nsize
        include 'PPLDAT.INC'       ! with nsize
        include 'switch_inc.decl'
        include 'SWITCH.INC'

*kob 1/97
	include 'xwindow_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPPLUS.CMN'
	INCLUDE 'FERRET_CMN:XBOX_BOUNDS.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'	! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'	! axis lengths
        INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL/LIST'       ! with nsize
        INCLUDE 'PPLV11INC:PPLDAT.INC/LIST'       ! with nsize
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
#endif


* calling argument declarations:
	LOGICAL		vector, overlay, transpz, no_labels, no_x_rng
	INTEGER		nmv, mr_list( nmv ), cx_list( nmv ), uvar,
     .			pen, status
	REAL		memory( mem_blk_size, max_mem_blks )
	REAL 		aspect, valx1, s1_axis(*), s2_axis(*)

* internal variable declarations:
	CHARACTER*200	COMPOUND_TITLE, COMPOUND_KEY, plot_title
	CHARACTER	TM_FMT*16, LEFINT*8,
     .                  xstyle*3, ystyle*3, cmnd_name*8, year*4, buff*8,
     .			letter*1
	LOGICAL		NO_GRID_RANGE, GEOG_LABEL, user_title, time_axis,
     .			regrid_1, regrid_2, no_y_rng, formatted,
     .			edges_ok
	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_SIZE,
     .			dim1, dim2, mvx, mvx_temp, mvy, mvy_temp,
     .			ctx, cty, xblk1, yblk1, plot_mem_used, nvars,
     .			lo1, lo2, hi1, hi2, size1, size2, size,
     .			n1_regrid, n2_regrid, grid, slen, pxlim, pylim,
     .			style, justify, ipen, ndim, dims(4), ptitl,
     .			arg_xpos, cx_xpos, cx_ypos, mr_xpos, mr_ypos
	REAL		s, w1_lo, w1_hi, w2_lo, w2_hi, valy1,
     .			d1_regrid, d2_regrid, bad_x, bad_y, rlo, rhi,
     .			xkeypos, ykeypos, xkeysize, ykeysize, lab_ht
	REAL*8		xaxlo, xaxhi, yaxlo, yaxhi, delta

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			'setting up 2D plot', 18 )

* initialize
	mvx	  = mr_list( 1 )
	ctx       = cx_list  ( 1 )
	grid	  = mr_grid( mvx )
	bad_x	  = mr_bad_data( mvx )
	IF ( vector ) THEN
	   mvy	  = mr_list( 2 )
	   cty	  = cx_list  ( 2 )
	   bad_y  = mr_bad_data( mvy )
	ENDIF
	IF ( transpz ) THEN
	  pxlim = qual_given( slash_ylimits )
	  pylim = qual_given( slash_xlimits )
	ELSE
	  pxlim = qual_given( slash_xlimits )
	  pylim = qual_given( slash_ylimits )
	ENDIF
	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0

* determine PLOT+ memory required
        plot_mem_used = CGRID_SIZE( cx_list(1) )
        IF ( vector ) plot_mem_used = plot_mem_used * 2
        IF ( plot_mem_used .GT. 0.99*nsize ) GOTO 5170

* set flag indicating a 2D plot is on the screen
	twodee_on = .TRUE.

* get 4D shape of data region to load (already oversized)
	dim1 = plot_axis(1)
	dim2 = plot_axis(2)
	size1 = CX_DIM_LEN( dim1, ctx )
	size2 = CX_DIM_LEN( dim2, ctx )
	size  = size1 * size2

* save axis pointers so PLOT+ can get box boundaries
	bb_grid = grid
	IF ( transpz ) THEN
	   bb_yaxis = dim1
	   bb_xaxis = dim2
	ELSE
	   bb_xaxis = dim1
	   bb_yaxis = dim2
	ENDIF

* check for valid data plane
	CALL GET_CX_DIMS( ctx, ndim, dims )
	IF ( overlay ) THEN
* *sh* 1/93 - restrictions less: coords merely have to be valid in 2D plane
           IF ( cx_lo_ss(ctx,dim1) .EQ. unspecified_int4
     .     .OR. cx_lo_ww(dim1,ctx) .GE. cx_hi_ww(dim1,ctx) ) GOTO 5150
           IF ( cx_lo_ss(ctx,dim2) .EQ. unspecified_int4
     .     .OR. cx_lo_ww(dim2,ctx) .GE. cx_hi_ww(dim2,ctx) ) GOTO 5150
	ELSEIF ( vector) THEN     ! 1/94
	   IF ( nplot_axis .NE. 2 ) GOTO 5160
	ELSE
	   IF ( ndim .NE. 2 ) GOTO 5160
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
	ENDIF
	lo1  = cx_lo_ss( ctx, dim1 )
	hi1  = cx_hi_ss( ctx, dim1 )
	lo2  = cx_lo_ss( ctx, dim2 )
	hi2  = cx_hi_ss( ctx, dim2 )
	size1 = hi1 - lo1 + 1
	size2 = hi2 - lo2 + 1

* check that there is some valid data
	no_x_rng = NO_GRID_RANGE( memory(1,xblk1), size, mvx_temp, valx1 )
	IF ( no_x_rng .AND. valx1 .EQ. bad_x ) GOTO 5200
	IF ( vector ) THEN
	   no_y_rng = NO_GRID_RANGE( memory(1,yblk1), size, mvy_temp, valy1 )
	   IF ( no_y_rng .AND. valy1 .EQ. bad_y  ) GOTO 5200
	ENDIF

* determine axis re-gridding needs
	CALL AXIS_INTERVALS( 'X', mvx_temp, ctx, dim1, curvilinear, s1_axis,
     .			     mr_lo_ss(mvx_temp,dim1), mr_hi_ss(mvx_temp,dim1),
     .			     w1_lo, w1_hi, n1_regrid, d1_regrid, regrid_1 )
	CALL AXIS_INTERVALS( 'Y', mvx_temp, ctx, dim2, curvilinear, s2_axis,
     .			     mr_lo_ss(mvx_temp,dim2), mr_hi_ss(mvx_temp,dim2),
     .			     w2_lo, w2_hi, n2_regrid, d2_regrid, regrid_2 )
	IF ( regrid_1 .OR. regrid_2 ) THEN
	   icode = 1
	ELSE
	   icode = 0
	ENDIF

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

* set up the axis formats (use exact limits, if given by /XLIMITS or /YLIMITS)
	IF ( .NOT.overlay ) THEN
	   IF ( pxlim .GT. 0 ) THEN
	      CALL EQUAL_RANGE_T( cmnd_buff(qual_start(pxlim):qual_end(pxlim)),
     .				dim1, grid, xaxlo, xaxhi, delta,
     .				status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( curvilinear ) THEN
	      CALL FIELD_MINMAX( memory(1,mr_blk1(mr_xpos) ), 
     .				mr_xpos, cx_xpos, rlo, rhi )
	      xaxlo = rlo
	      xaxhi = rhi
	      IF (xaxlo .EQ. xaxhi) THEN
		 letter = 'X'
		 GOTO 5500
	      ENDIF
	      delta = unspecified_val8
	   ELSE
	      CALL CHOOSE_CX_WW( uvar, cx_cmnd, ctx, dim1, xaxlo, xaxhi )
	      delta = unspecified_val8
	   ENDIF
	   haxlo = xaxlo	! save limits so overlays
	   haxhi = xaxhi	! will know them
	   time_axis = GEOG_LABEL( dim1, grid )
	   CALL AXIS_ENDS  ( 'X', dim1, grid, xaxlo, xaxhi, delta )
	   CALL AXIS_FORMAT( 'X', dim1, grid, xaxlo, xaxhi,
     .						curvilinear, cx_xpos,
     .						time_axis, xstyle)
	   IF ( pylim .GT. 0 ) THEN
	      CALL EQUAL_RANGE_T( cmnd_buff(qual_start(pylim):qual_end(pylim)),
     .				dim2, grid, yaxlo, yaxhi, delta,
     .				status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( curvilinear ) THEN
	      CALL FIELD_MINMAX( memory(1,mr_blk1(mr_ypos) ),
     .				mr_ypos, cx_ypos, rlo, rhi )
	      yaxlo = rlo
	      yaxhi = rhi
	      IF (yaxlo .EQ. yaxhi) THEN
		 letter = 'Y'
		 GOTO 5500
	      ENDIF
	      delta = unspecified_val8
	   ELSE
	      CALL CHOOSE_CX_WW( uvar, cx_cmnd, ctx, dim2, yaxlo, yaxhi )
	      delta = unspecified_val8
	   ENDIF
	   vaxlo = yaxlo	! save limits so overlays
	   vaxhi = yaxhi	! will know them
	   time_axis = GEOG_LABEL( dim2, grid )
	   CALL AXIS_ENDS  ( 'Y', dim2, grid, yaxlo, yaxhi, delta )
	   CALL AXIS_FORMAT( 'Y', dim2, grid, yaxlo, yaxhi, 
     .						curvilinear, cx_ypos,
     .						time_axis, ystyle )
        ENDIF

* rotate vectors to make axis scale ratio correction ?
	IF ( vector .AND. aspect.GT.0. ) CALL RE_ORIENT(
     .			haxlo, haxhi, vaxlo, vaxhi, dim1, dim2,
     .			size1, size2, transpz, aspect,
     .			memory(1, xblk1), mvx_temp,
     .			memory(1, yblk1), mvy_temp, s )

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

	if (overlay) SOVER = .true.

* load contour data into PPLUS and (possibly) regrid to regular grid
	CALL PPLLDC ( icode, memory(1, xblk1), size1, size2,
     .		      1, size1, 1, size2, s1_axis, s2_axis, 
     .		      n1_regrid, n2_regrid, w1_lo, w2_lo,
     .		      d1_regrid, d2_regrid )
	CALL DELETE_VARIABLE( mvx_temp )

	IF ( vector ) THEN
* load the Y vector component
	   CALL PPLLDV ( icode, memory(1,yblk1), size1, size2,
     .		      1, size1, 1, size2			)
	   CALL DELETE_VARIABLE( mvy_temp )
	ENDIF
	
	IF ( no_labels ) RETURN

! ====================== LABELS ========================

* if we are starting from a blank screen compose a title
* if we are overlaying on an plot put the new variable name on as a key
	IF ( user_title ) THEN
	   CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   slen = TM_LENSTR1( plot_title )
	ENDIF
	IF ( overlay ) THEN
* ... cook up a key label for this variable only (or pair for VECTOR)
	   use_keys = .TRUE.
	   IF ( .NOT.user_title ) THEN
	      CALL OV_GRP_CX_DOC( cx_list, nmv, cx_xtra_doc )
	      plot_title = COMPOUND_KEY( cx_list, nvars, cx_xtra_doc, slen )
	   ENDIF
	   CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, justify )
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
	      CALL TSTEP_TO_DATE( grid, xaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ELSEIF (dim2.EQ.t_dim.AND.
     .                  (ystyle.EQ.'MON'.OR.ystyle.EQ.'DAY')) THEN
	      CALL TSTEP_TO_DATE( grid, yaxlo, ptime_yr, ppl_buff )
	      year = ppl_buff(:4)
	   ENDIF
* ... put on the basic position and data set labels
	   CALL DISP_LABELS( cx_plot, year )
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
* ... offset components ?
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
 5160	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'can only contour or vector a 2D region'//pCR//
     .					cmnd_buff(:len_cmnd), *5100 )
 5170   buff = LEFINT( INT(0.99*nsize), lo1 )
        cmnd_name = LEFINT( plot_mem_used, lo2 )   ! used as buffer
        CALL ERRMSG( ferr_prog_limit, status,
     .          cmnd_name(:lo2)//' points to plot'//pCR//
     .          'Plot buffer size is: '//buff(:lo1), *5100 )

 5200	CALL ERRMSG( ferr_no_range, status,
     .				'data are all flagged as bad', *5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		     'can only '//commands(cmnd_num)//' single fields'//pCR//
     .		     'Use /OVERLAY for multiple variables', *5100 )
 5400	CALL ERRMSG( ferr_inconsist_plane, status,
     .		 letter//' coord field lies'//pCR//
     .		'in different plane from data to be plotted', *5100 )
 5500	CALL ERRMSG( ferr_no_range, status, letter//
     .				' coordinate field', *5000 )
	END
