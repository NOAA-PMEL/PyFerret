	SUBROUTINE PLOT_SET_UP( memory,
     .				overlay, transpz, versus, no_labels,
     .				symbol, sym_size, color, use_line,
     .				mv_list, cx_list, nmv,
     .				indep_dat, dep_dat, status )

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
* set up for line plot by loading data and title and defining axis formats
* (the routine is confusing as a result of simultaneously coping with the needs
* of single variables, multiple variables, /VS, /TRANSPOSE and /OVERLAY)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87  - based on DISP_SET_UP rev 1.1
* revision 0.1 - 7/21/87 - added transpose option and AXIS_ENDS
* revision 0.2 - 8/26/87 - added TAXIS ON for time axis plots
* revision 0.3 - 2/29/88 - allowed for non-geometric axes, time axis formats,
*			   non-inverted Z data,
*			   and check to keep PLOT+ from screwing up when data 
*			   are all a single value
* revision 0.4 - 4/12/88 - added do_labels argument for "/NOLABELS" option
*			   and units labeled on dependent axis
* revision 0.5 - 4/28/88 - force-scale axes if data is all 1 value
* revision 0.6 -  9/8/88 - added relative flags to PUT_LABEL calls
* revision 1.0 - 9/23/88 - allow lines of different lengths - bypass time axis
*			   limits as computed by PPLLDX because of bugs in that
*			   code.  Move /no_labels to PUT_LABEL.
*			 - put "year" label on if time axis doesn't
* V200:  5/19/89 - reordered lo/hi_ss arguments
*         1/8/89 - major rewrite using keys
*	  5/8/90 - allow PLOT/VS to overlay 2D fields
*		 - PLOT/VS needn't be given a line - it can be a higher dim reg.
*		 - improved error message for plot not of a line
*	 5/17/90 - aded PLOT/SYMBOL and PLOT/LINE
*  Unix/RISC port 3/91 *sh* : bug: TSTEP_TO_SECS requires REAL*8 arg
*                 8/23/91 ; bug when time axis is normal cannot find dt_min
* V230:   6/8/92 - corrected bug when line_t0 and line_units may be undefined
*        6/12/92 - trap PLOT+ nsize limitation before passing data
*        6/26/92 - time_axis variable not properly init'ed for /vs plot
*        11/4/92 - expanded variable name length
*                - corrected small bug in size of dependent axis label
*       11/27/92 - symptomatic bug fix of mis-scaled TAXIS0 (hours) axis
*                - see err230_ppl_taxis.jnl
* V300: 3/11/93  - allow free-er overlay of plot on top of 2D display
* V301:	10/22/93 - added /XLIMITS and /YLIMITS  (variables dep_min &_max
*		   commandeered as not really used earlier)
*	 11/3/93 - added /TITLE
*	  1/7/94 - screen bad data flag on independent axis only for /VS
* V320: 5/94 - array "memory" as a calling argument
* 	9/94: restrict to 99% of NSIZE buffer - kludge fix to PLOT+ bug
*       12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420: 11/27/95 - 4-digit PLOT+ YEARS!!
*	11/28/95 - changes to TAXIS_STYLE to support y-oriented calendar axes
*	 3/26/96 - number of expressions to be merged was incorrect in call to
*			MERGE_WHOI_CONTEXT
* V450: 3/97 - modified AXIS_FORMAT args for curvilinear coordinate plots
* V500 *kob* 3/99- up VAR_CODE  and VAR_UNITS to 64 chars
*       4/99 - *jd* Ensure PPL knows plot is overlay: set SOVER flag here
*       6/99 - *sh* added logic to permit a PLOT/VS over a time series plot
* V510 10/99 - *sh* plot/i=1:10 i/0 is not an error -- scale axis to -1:1
*       3/00 - *sh* added use_line and sym_size args for PLOT/COLOR=/SIZE=
*	     - pass no_labels to AXIS_FORMAT 

#ifdef unix
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
        include 'axis_inc.decl'       ! with axis lengths
        include 'AXIS.INC'       ! with axis lengths	
        include 'ppldat_inc.decl'       ! with nsize
        include 'PPLDAT.INC'       ! with nsize
        include 'switch_inc.decl'
        include 'SWITCH.INC'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
        INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL/LIST'       ! with axis lengths
        INCLUDE 'PPLV11INC:AXIS.INC/LIST'       ! with axis lengths	
        INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL/LIST'       ! with nsize
        INCLUDE 'PPLV11INC:PPLDAT.INC/LIST'       ! with nsize
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
#endif

* calling argument declarations:
	LOGICAL		overlay, transpz, versus, no_labels, use_line
	INTEGER		symbol, color,
     .			nmv, mv_list( nmv ), cx_list( nmv ), status
	REAL		memory( mem_blk_size, max_mem_blks ), sym_size
	REAL		indep_dat(*), dep_dat(*)     ! dynamic space allocation
 
* V500 *kob* 3/99- up VAR_CODE and VAR_UNITS to 64 chars
* internal variable declarations:
	LOGICAL		NO_LINE_RANGE, GEOG_LABEL, flip, time_axis,
     .			all_1_dep, all_1_ind, formatted, user_title
	CHARACTER*200	FULL_VAR_TITLE,  KEY_STRING, plot_title
	CHARACTER	INTERNAL_WHOI_DATE*14, VAR_UNITS*64,
     .			SECS_TO_DATE*20, LEFINT*4, MERGED_WHOI_DATE*14,
     .			indep_ax*1, dep_ax*1, tstyle*3, year*4,
     .			buff1*8, buff2*8, buff3*8, t1_date*14, tref*14
	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_SIZE,
     .			mv, cx, slen, ito, indep_lab, dep_lab, nline_in_mem,
     .			indep_dim, npts, dep_dim, npts2, ndv, var1,
     .			grid, grid1, ipl, style, justify, ndim, dims(4),
     .			slen2, slen3, tax, plot_mem_used,
     .			pxlim, pylim, ptitl
	REAL		only_val, val1, dt_min,
     .			lo, hi, dep_len, ind_len,
     .			xkeypos, ykeypos, xkeysize, ykeysize, lab_ht
	REAL*8		TSTEP_TO_SECS, SECS_TO_TSTEP,
     .			ind_min, ind_max, dep_min, dep_max, delta, tmp

* local parameter declarations:
	LOGICAL		main, norm_labs, pnot_curv
	PARAMETER     ( main = .TRUE.,
     .			norm_labs = .FALSE.,
     .			pnot_curv = .FALSE. )

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'setting up plot', 15)

* initialize
!	dep_max = arbitrary_small_val8
!	dep_min = arbitrary_large_val8
	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	cx   = cx_list( 1 )
	mv   = mv_list( 1 )
	grid1 = cx_grid( cx )
	indep_dim = plot_axis(1)
	nline_in_mem = 0
	pxlim = qual_given( slash_xlimits )
	pylim = qual_given( slash_ylimits )
	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0

* determine PLOT+ memory required
        plot_mem_used = 0
        DO 10 ipl = 1, nmv
 10     plot_mem_used = plot_mem_used + CGRID_SIZE( cx_list(ipl) )
        IF ( plot_mem_used .GT. 0.99*nsize ) GOTO 5150

* check for improper data supplied
	IF ( versus ) THEN
	   IF ( num_uvars_in_cmnd .LT. 2 ) GOTO 5100
	ELSEIF ( overlay ) THEN
	   CALL GET_CX_DIMS( cx, ndim, dims )
           IF ( twodee_on ) indep_dim = dims(1)       ! 3/93
	   IF ( ndim.NE.1 .OR. dims(1).NE.indep_dim ) GOTO 5120
	ELSE
	   ndim = nplot_axis
	   IF ( ndim .GT. 1 ) GOTO 5110
	ENDIF

* length of plot array
	IF ( versus ) THEN
	   npts = CGRID_SIZE( cx )	! (redundantly calc'd below) 5/90
	ELSE
	   npts = CX_DIM_LEN ( indep_dim, cx )
	ENDIF

* set flag indicating a 1D plot is on the screen
	IF (.NOT.overlay) onedee_on = .TRUE.

* decide if the picture should be rotated 90 degrees
	IF ( .NOT.versus .AND. indep_dim .EQ. z_dim  ) THEN
* ... flip depth plots by default
	   flip = .NOT.transpz
	ELSE
	   flip = transpz
	ENDIF
	IF ( flip ) THEN
	     dep_ax  = 'X'
	   indep_ax  = 'Y'
	     dep_lab = ppl_xlab
	   indep_lab = ppl_ylab
	   dep_len   = xlen
	   ind_len   = ylen
	ELSE
	     dep_ax  = 'Y'
	   indep_ax  = 'X'
	     dep_lab = ppl_ylab
	   indep_lab = ppl_xlab
	   ind_len   = xlen
	   dep_len   = ylen
	ENDIF

* * * * * * *    - - - INDEPENDENT AXIS DATA - - - * * * * * * * * *
	IF ( versus ) THEN
	   time_axis =	.FALSE.
	   CALL EXTRACT_LINE ( cx,
     .			    memory( 1, mr_blk1(mv) ),
     .			    mv,
     .			    indep_dat,
     .			    indep_dim,
     .			    ndim,
     .			    npts,
     .			    status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   var1 = 2	! since first var is independent axis
	   ndv  = nmv - 1	! # of dependent variables
	   IF ( use_keys ) THEN
	      plot_title = KEY_STRING( cx_list(1), key_doc, slen )
	   ELSE
	      plot_title = FULL_VAR_TITLE( cx_list(1), slen )
	   ENDIF
* ... label the independent axis as the first var given
	   IF ( .NOT.no_labels ) THEN
	   CALL BOX_LABEL( indep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*ind_len, letsize_label,
     .			   letsize_label, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	   ENDIF
* ... force axis scaling if the data has no range
	   all_1_ind = NO_LINE_RANGE( indep_dat,npts,mr_bad_data(mv),val1 )
	   delta = unspecified_val8
	   IF ( pxlim .GT. 0 ) THEN   ! explicit /XLIMITS 10/93
	      CALL EQUAL_RANGE( cmnd_buff(qual_start(pxlim):qual_end(pxlim)),
     .				indep_dim, ind_min, ind_max, delta,
     .				formatted, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( all_1_ind .AND. .NOT.overlay ) THEN
	      IF ( val1 .EQ. mr_bad_data(mv) ) THEN
	         ind_min = 0.D0
	         ind_max = 1.D0
	      ELSE
	         ind_min = val1 - 1.
	         ind_max = val1 + 1.
	      ENDIF
	   ELSE
	      CALL MINMAX( indep_dat, npts, mr_bad_data(mv), lo, hi )
	      ind_min = DBLE( lo )
	      ind_max = DBLE( hi )
	   ENDIF
	   IF (.NOT.overlay) THEN
	      CALL AXIS_ENDS(indep_ax,indep_dim,grid1,ind_min,ind_max,delta)
* ... turn off axis formatting for /VS plots
	      CALL PPLCMD ( from, line, 0, indep_ax//'FOR 0', 1, 1 )
	   ENDIF
	ELSE
	   var1 = 1
	   ndv  = nmv
	   ito  = 0
	   time_axis =	indep_dim.EQ.t_dim
	   IF ( time_axis ) THEN	! 11/95: are ALL vars time vars?
	      DO 100 ipl = var1, nmv
	         time_axis = time_axis
     .			.AND. GEOG_LABEL( t_dim, cx_grid(cx_list(ipl)) )
 100	      CONTINUE
	   ENDIF
* ... (PLOT+ needs TAXIS,ON prior to PPLLDX calls)
* ... PPLLDX computes axis endpoints if TIME is set to auto (no args given)
	   IF ( time_axis ) THEN
! *sh* 11/92 added numerical dt_min argument to TAXIS call in PLOT+
! this fixes err230_ppl_taxis.jnl symptomatically but probably leaves an
! unresolved scaling problem - especially with multiple time series w/ diff
! units so that the dt_min argument is not approproate to subsequent lines
! problem seems to go away if DT=60.0
	      CALL PPLCMD ( from, line, 0, 'TIME', 1, 1 )
	      IF ( indep_ax .EQ. 'X' ) THEN
	         CALL PPLCMD ( from, line, 0, 'TAXIS 60,ON', 1, 1 )
	      ELSE
	         CALL PPLCMD ( from, line, 0, 'TAXIS/YAXIS 60,ON', 1, 1 )
	      ENDIF
	   ENDIF
	ENDIF

* set up PLOT5 to ignore bad data flag
	IF ( versus ) THEN               ! 1/94
	   ppl_buff = ' '
	   WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), indep_ax
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ENDIF
 3005	FORMAT ( 'LIMITS ',G15.8,1X,A1,'EQ' )

* * * * * * *    - - - LOAD DATA - - - * * * * * * * * *
* 11/95: determine T0 reference time for time axis
	IF ( time_axis .AND. .NOT.overlay ) THEN
	   tref = MERGED_WHOI_DATE( cx_list(var1), nmv )
	ELSE
	   tref = ' '		! overlays reuse previous tref
	ENDIF

* loop through all the dependent variables loading them into PPLUS
	all_1_dep = .TRUE.
	only_val  = bad_val4
	DO 200 ipl = var1, nmv
	   mv   = mv_list( ipl )
	   cx   = cx_list( ipl )
	   grid = cx_grid( cx )
* set up a dependent variable
	   CALL EXTRACT_LINE (	cx,
     .				memory( 1, mr_blk1(mv) ),
     .				mv,
     .				dep_dat,
     .				dep_dim,
     .				ndim,
     .				npts2,
     .				status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000

* set up corresponding independent axis
* ( note - each variable may have different points on the indep. axis )
	   IF ( versus ) THEN
* ... number of values must be equal in independ. and depend. for ordered pairs
	      IF ( npts2 .NE. npts ) GOTO 5130
	   ELSE
* ... in PLOT TEMP , SALT , U all vars must be along same ww dimension
	      IF ( ndim .GT. 1 ) GOTO 5110
	      IF ( dep_dim .NE. indep_dim ) GOTO 5140
	      CALL MAKE_AXIS( grid, dep_dim,
     .			cx_lo_ss(cx,dep_dim), cx_hi_ss(cx,dep_dim),
     .			indep_dat )
	      IF ( time_axis ) THEN
	         ind_min = MIN( ind_min,
     .                     TSTEP_TO_SECS(grid, DBLE(indep_dat(1)) ))
	         ind_max = MAX( ind_max,
     .                     TSTEP_TO_SECS(grid,DBLE(indep_dat(npts2)) ))
	      ELSE
	         ind_min = MIN( ind_min, DBLE(indep_dat(1)) )
	         ind_max = MAX( ind_max, DBLE(indep_dat(npts2)) )
	      ENDIF
	      all_1_ind = ind_min .EQ. ind_max
	      IF ( all_1_ind ) THEN
	         ind_min = ind_min - 1.
	         ind_max = ind_max + 1.
	      ENDIF
	   ENDIF

* ... check that there is a range of dependent data for PLOT+ auto-scaling
* ... when all the variables are considered together
	all_1_dep = all_1_dep .AND. 
     .		NO_LINE_RANGE( dep_dat, npts2, mr_bad_data(mv), val1 )
! 10/99	IF ( val1 .EQ. mr_bad_data(mv) ) CALL ERRMSG( ferr_no_range, status,
!     .		'all data points are bad: '//VAR_CODE(mr_category(mv),
!     .						      mr_variable(mv) ),*5000)
	IF ( all_1_dep ) THEN
	   IF ( only_val .EQ. bad_val4 ) THEN
	      IF ( val1 .NE. mr_bad_data(mv) ) only_val = val1
	   ELSE
	      all_1_dep = all_1_dep .AND. only_val .EQ. val1
	   ENDIF
!	ELSE  ! commented out 10/93 - PLOT+ auto scaling used
!	   CALL MINMAX( dep_dat, npts, mr_bad_data(mv), lo, hi )
!	   IF ( lo .LT. dep_min ) dep_min = lo
!	   IF ( hi .GT. dep_max ) dep_max = hi
	ENDIF

* pass the data to PLOT+
	   WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), dep_ax
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   IF ( time_axis ) THEN
              tax = grid_line(t_dim,grid)
              IF ( tax .EQ. mnormal .OR. tax .EQ. munknown ) THEN  ! 8/91 bug
                 dt_min = 1.0
              ELSE
	         dt_min = line_tunit( grid_line(t_dim,grid) ) / 60.! sec-->min
              ENDIF
              t1_date = INTERNAL_WHOI_DATE( grid, 1.0D0 )
           ELSE
	      IF (overlay .AND. has_time_axis) THEN
	         dt_min  = saved_dt_min
	         t1_date = saved_t1_date
	      ELSE
                 dt_min  = 1.0
                 t1_date = ' '
	      ENDIF
           ENDIF

	   IF (overlay) THEN
	      SOVER = .true.
	   ELSE
	      has_time_axis = time_axis
 	      IF (has_time_axis) THEN
	         saved_dt_min  = dt_min		! to re-use on PLOT/VS
	         saved_t1_date = t1_date
	      ENDIF
	   ENDIF

	   IF ( flip ) THEN
	      CALL PPLLDX(0,dep_dat,indep_dat,npts2,t1_date,tref,dt_min)
	   ELSE
	      CALL PPLLDX(0,indep_dat,dep_dat,npts2,t1_date,tref,dt_min)
	   ENDIF
* ... increment number of lines on plot
	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1
* ... assign line style for the data plotting:
	   CALL LINE_STYLE(symbol, sym_size, color, use_line,
     .			   nline_in_mem,nline_on)
 200	CONTINUE

* axis scaling and formatting
* ... independent axis
	IF ( .NOT.overlay ) THEN
	   IF ( .NOT. versus ) THEN
	      delta = unspecified_val8
	      IF ( pxlim .GT. 0 ) THEN              ! explicit /XLIMITS 10/93
	         CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pxlim):qual_end(pxlim)),
     .			indep_dim, ind_min, ind_max, delta,
     .			formatted, status )
	         IF ( status .NE. ferr_OK ) GOTO 5000
* convert time given in time steps to absolute units	   
	         IF ( time_axis .AND. .NOT.formatted ) THEN
* time limits given as tstep range although mode calendar is true
	            IF ( ind_min .GT. ind_max ) THEN  ! no reversed calendar
	               tmp = ind_min
	               ind_min = ind_max
	               ind_max = tmp
	            ENDIF
	            ind_min = TSTEP_TO_SECS( grid1, ind_min )
	            ind_max = TSTEP_TO_SECS( grid1, ind_max )
	         ELSEIF ( indep_dim .EQ. t_dim
     .		   .AND. .NOT.time_axis
     .	           .AND.  formatted ) THEN
* time limits given as dates although mode calendar is false
	            ind_min = SECS_TO_TSTEP( grid1, ind_min )
	            ind_max = SECS_TO_TSTEP( grid1, ind_max )
	         ENDIF
	      ENDIF
	      IF ( time_axis ) THEN
	         IF ( pxlim .GT. 0 ) CALL PPL_TIME( ind_min,ind_max )
	         CALL TAXIS_STYLE( indep_ax, ind_min, ind_max, tstyle )
* (6/99) Note: The time axis extrema symbols are only approximately correct
*		since PLOT+ rounds out the time ranges ...
	         CALL AXIS_END_SYMS( indep_ax,
     .				SECS_TO_TSTEP( grid1, ind_min ),
     .				SECS_TO_TSTEP( grid1, ind_max ) )
	      ELSE
	         CALL AXIS_ENDS  ( indep_ax, indep_dim, grid1,
     .				   ind_min, ind_max, delta )
	         CALL AXIS_FORMAT( indep_ax, indep_dim, grid1,
     .				   ind_min, ind_max,
     .				   pnot_curv, 0,
     .				   time_axis, tstyle, no_labels )
	      ENDIF
	   ENDIF
* ... dependent axis scaling
*  force axis scaling if the data has no range
	   IF ( pylim .GT. 0 ) THEN                 ! explicit /YLIMITS 10/93
	      CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pylim):qual_end(pylim)),
     .			dep_dim, dep_min, dep_max, delta,
     .			formatted, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	      CALL AXIS_ENDS( dep_ax, t_dim, grid, dep_min, dep_max, delta )
	   ELSEIF ( all_1_dep ) THEN
	      IF (only_val .EQ. bad_val4) val1 = 0.0	! 10/99
	      CALL AXIS_ENDS( dep_ax, t_dim, grid,val1-1.D0,val1+1.D0,1.D0 )
	   ENDIF
	ENDIF


	IF ( no_labels ) RETURN

* * * * * * *    - - - TITLES - - -   * * * * * * * * *
* if we are starting from a blank screen titling information is centered
* below the X axis - either a title for a single variable or keys for multiple
* if we are overlaying on an existing plot put the new variable name on as a
* key at the lower left
	nline_in_mem = 0
	IF ( overlay ) THEN
	   DO 500 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      CALL OV_GRP_CX_DOC( cx_list(ipl), 1, cx_xtra_doc )
	      IF ( user_title ) THEN
	         CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         slen = TM_LENSTR1( plot_title )
	      ELSE
	         plot_title = KEY_STRING( cx_list(ipl), cx_xtra_doc, slen )
	      ENDIF
	      CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, justify )
	      CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			    xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, lab_loc_absolute, lab_loc_absolute )
 500	   CONTINUE
	ELSE
* ... basic position and data set labels
	   IF ( tstyle.EQ.'MON' .OR. tstyle.EQ.'DAY' ) THEN
	      ppl_buff = SECS_TO_DATE( ind_min )
	      year = ppl_buff(8:11)
	   ELSE
	      year = unspecified_name4
	   ENDIF
	   CALL DISP_LABELS( cx_plot, year )
* ... label dependent axis with units (over-written below if /vs)
	   plot_title = VAR_UNITS( cx_list(var1) )
	   slen = TM_LENSTR1( plot_title )
	   CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label,
     .			   letsize_label, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
* ... keys for the dependent vars, or if one, only then title or axis label
	   DO 600 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      IF ( use_keys ) THEN
	         plot_title = KEY_STRING( cx_list(ipl), key_doc, slen )
	      ELSE
	         plot_title = FULL_VAR_TITLE( cx_list(ipl), slen )
	      ENDIF
	      IF ( ndv .EQ. 1 .OR. user_title ) THEN
	         IF ( versus ) THEN     ! ("xlen" changed to "dep_len" 11/92)
	            CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label,
     .			   letsize_label, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         ENDIF
	         IF ( .NOT.versus .OR. user_title ) THEN 
	            IF ( user_title ) THEN     ! user over-ride of title
	               CALL EQUAL_STR_LC
     .			( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                    plot_title, status )
                       IF ( status .NE. ferr_ok ) GOTO 5000
	               slen = TM_LENSTR1( plot_title )
	            ENDIF
	            CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*xlen, letsize_title,
     .			   letsize_title, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         ENDIF
	      ELSE
	         CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			       lab_ht, style, justify )
	         CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			       xkeypos, ykeypos, xkeysize, ykeysize,
     .			       lab_ht, style,
     .			       lab_loc_absolute, lab_loc_absolute )
	      ENDIF

 600	   CONTINUE
	ENDIF

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status, 
     .		     cmnd_buff(:len_cmnd)//' : vs what ?', *5000 )
 5110	dep_ax = LEFINT( ndim, slen )
	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'specified data is not a line'//pCR//
     .			' - its a '//dep_ax//'D region: "'//
     .			cmnd_buff(:len_cmnd)//'"', *5000 )
 5120	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'overlay is on a different axis'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
 5130	buff1 = LEFINT( npts,  slen  )
	buff2 = LEFINT( npts2, slen2 )
	buff3 = LEFINT( ipl, slen3 )
	CALL ERRMSG( ferr_dim_underspec, status,
     .		 'unequal line lengths: '//pCR//
     .		'First expression has '//buff1(:slen)//' points.'//pCR//
     .		'Expression '//buff3(:slen3)//' has '//buff2(:slen2)//
     .						' points:'//
     .		pCR//'"'//cmnd_buff(:len_cmnd)//'"', *5000 )

 5140	buff3 = LEFINT( ipl, slen3 )
	CALL ERRMSG( ferr_dim_underspec, status,
     .		'differing axes: '//pCR//
     .		'first line is on '//ww_dim_name(indep_dim)//' axis'//pCR//
     .		'line '//buff3(:slen3)//' is on '//ww_dim_name(dep_dim)//
     .		' axis', *5000 )
        
 5150	buff3 = LEFINT( INT(0.99*nsize), slen3 )
	buff2 = LEFINT( plot_mem_used, slen2 )
	CALL ERRMSG( ferr_prog_limit, status,
     .          'Requested '//buff2(:slen2)//' words to plot'//pCR//
     .		'Plot buffer size is: '//buff3(:slen3), *5000 )
	END
