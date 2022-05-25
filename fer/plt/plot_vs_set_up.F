        SUBROUTINE PLOT_VS_SET_UP( 
     .				overlay, transpz, no_labels,
     .				symbol, sym_size, color, color1, use_line,
     .				step_inc, do_dash, dashstyle,
     .                          is_logx, is_logy, nokey, addgaps,
     .				all_1_dep, only_val, skipsym, mv_list, 
     .				cx_list, nmv, indep_dat, dep_dat, status )

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
* This version handles all of the logic for versus plots, pulling that out of
* plot_set_up.F
* 9/2020 ACM
* v763 *acm* 9/20 Fixing bug with PLOT/VS of single-valued data, negative longitudes.
* V7.64  7/2021 *acm* Add a little "padding" for the axis ends, so that the points do not lie exactly
*                     on the axes.
* V7.64  7/2021 *acm* for pyferret ticket #89: plot/vs command with /axes  after a plot/vs, lost the 
*                  longitude-axis formatting
* V766  5/22 *acm* Update the 1099/1888 changes for a PLOT/VS overlay plot on a time plot
*                  that has rescaled time.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
        include 'xunits.cmn_text'
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
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'PPLDAT.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
	include 'xdset_info.cmn_text'
	include 'xdsg_context.cmn'

#include "tmap_dset.parm"

* calling argument declarations:
	LOGICAL		overlay, transpz, no_labels, use_line,
     .			is_logx, is_logy, nokey, all_1_dep, addgaps
	INTEGER		symbol, color, color1, step_inc, do_dash, skipsym,
     .			nmv, mv_list( * ), cx_list( * ), status
	REAL		 sym_size,
     .                  dashstyle(*), only_val
	REAL		indep_dat(*), dep_dat(*)     ! dynamic space allocation
 
* V500 *kob* 3/99- up VAR_CODE and VAR_UNITS to 64 chars
* internal variable declarations:

	LOGICAL		NO_LINE_RANGE, GEOG_LABEL, ITS_FMRC,
     .                  ITSA_TRUEMONTH_AXIS, TM_HAS_STRING, TM_ITSA_DSG,
     .			versus, flip, time_axis,
     .			all_1_ind, formatted,
     .			indep_is_log, dep_is_log, use_nice,
     .                  this_no_range(max_line_on_plot), set_axis,
     .                  timeax_both, do_units, doing_ribbon_color,
     .                  x_is_time, y_is_time, its_traj, its_dsg, its_cmpnd,
     .                  expand_dsg_ok, color_dsg_id, user_hlim, dsg_as_time, 
     .                  dsg_as_traj, do_traj, native_traj, set_up_axes

	CHARACTER*2040	FULL_VAR_TITLE, KEY_STRING, plot_title
	CHARACTER	INTERNAL_WHOI_DATE*14, VAR_UNITS*64,
     .			SECS_TO_DATE*20, LEFINT*8, MERGED_WHOI_DATE*14, 
     .			LEFT_REAL*16, TM_GET_CALENDAR_NAME*32, 
     .			indep_ax*1, dep_ax*1, tstyle*3, xtra_lab*24,
     .			buff1*40, buff2*16, buff3*8, t1_date*14, tref*14,
     .                  cal_name*32, cal_name_new*32, val_buff*45, fmt*4,
     .			whoimin*14, whoimax*14, 
     .			whoimin_axlo*14, whoimax_axhi*14
	INTEGER*8	CGRID_SIZE8, gsize
	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_AXIS, 
     .			TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, TM_DSG_NF2FEATURES,
     .			mv, cx, slen, ito, indep_lab, dep_lab, nline_in_mem,
     .			indep_dim, npts, dep_dim, npts2, ndv, var1,
     .			grid, grid1, ipl, ndim, dims(6),
     .			slen2, slen3, tax, plot_mem_used, nload, i,
     .			pxlim, pylim, phlim, pvlim,pindeplim, pdeplim,
     .			dep_axtyp, indep_axtyp, the_taxis, ribbon_var, 
     .			gap_var, iiunits, iaxis, hblk1, mvh_temp, 
     .			nkey_entries, t_axtyp, since_T0, itmp, igrd_save, 
     .			icode, dset, nfeatures, obsdimlen, fline, orientation, 
     .			dtype, fvar, nload_all, numv, nok, line_symbol, 
     .			dsg_fmt_grid, dep_dir, nmasked, dset_dsg, dum1, dum2, 
     .			nftrsets 

	INTEGER         TM_GET_CALENDAR_ID, cal_id_1, cal_id_2,
     .			cal_id_old, cal_id_new
        INTEGER         TM_UNIT_ID, cx_x , cx_y, iunits, junits, axis_x_dir, axis_y_dir
	CHARACTER*1     axdir(6), ax1

	REAL		val1, dt_min, lo, hi, dep_len, 
     .			ind_len, first, last, hlen, hrs, 
     .			bc_min, bc_max, step_lo, step_hi, ascale, aoff
	REAL		TSTEP_TO_SECS, SECS_TO_TSTEP, DATE_TO_TSTEP, 
     .			WHOI2BC, GET_LINE_COORD, ind_min, ind_max, dep_min, dep_max, 
     .			delta, pad, tmp, dum, ww_save, cen, yr
	REAL*4 rbad

* local parameter declarations:
	LOGICAL		main, norm_labs, pnot_curv, range_rqd
	PARAMETER     ( main = .TRUE.,
     .			norm_labs = .FALSE.,
     .			pnot_curv = .FALSE.,
     .			range_rqd = .TRUE. )
	DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'setting up plot', 15)

	versus = .TRUE.

* limits error check
        IF (nmv .GT. max_line_on_plot) GOTO 5050
	
* initialize

	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	cx   = cx_list( 1 )
	mv   = mv_list( 1 )
	grid1 = cx_grid( cx )
	grid = cx_grid(is_cx(1))
	dset = cx_data_set(cx)
	indep_dim = plot_axis(1)
	twodee_on = .TRUE.

	numv = nmv  ! local copy
	line_symbol = symbol
	IF (cxdsg_empty_set) THEN
	   sym_size = 0.001
	   line_symbol = 1        ! doesn't matter what
	   use_line = .FALSE.
	ENDIF

* Set up axis properties, if not an overlay or if this an overlay with the /AXES qualifier, 
* in  order to draw the plot axes on the last plot

	set_up_axes = overlay .AND.   (qual_given(slash_plot_axes) .GT. 0 ) 

* ragged DSG data?  Get number of features and obsdimlen. If it's a 
* Trajectory or Point dataset and they did "PLOT var", set that up.

	 nfeatures = 0
	 its_dsg = .FALSE.
	 IF (dset .EQ. pdset_irrelevant) THEN
	    cx = is_cx(nmv)
	    grid1 = cx_grid( cx )
	    dset = cx_data_set(cx)
	 ENDIF

* a user-var or expression where the grid is dsg?

	 IF (dset .GT. pdset_irrelevant) its_dsg = dsg_ragged(dset)

	 dset_dsg = TM_DSG_DSET_FROM_GRID( grid )
	 IF (dset_dsg .GT. pdset_irrelevant) THEN
	    IF (.NOT. its_dsg) dset = dset_dsg
	    its_dsg = dsg_ragged(dset_dsg)
	 ENDIF
	 
	 IF (its_dsg ) THEN
	    igrd_save = grid
	    grid = cx_grid(is_cx(nmv))
	    IF (dset_dsg .LT. pdset_irrelevant) grid = dsg_xlate_grid(dset)

	    nfeatures = TM_DSG_NFEATURES( grid )	    

* The dataset may be DSG but the variable not.
	    CALL TM_DSG_FACTS( grid, orientation, obsdimlen, fline, its_dsg, its_cmpnd )
	    
	    nftrsets = TM_DSG_NF2FEATURES (dset)
	    
	    native_traj = orientation.EQ.pfeatureType_Trajectory .OR.  
     .	                  orientation.EQ.pfeatureType_Point

* Is this a PLOT/along=xy for some other dataset type?
* A bit convoluted: do_traj when plotting DSG as an XY map plot for any reason
* CHECK_PLOT_TRAJ is testing for a scatter plot based on a dsg dataset
* Then the XY map plot is called with a flag its_traj for whether it's 
* trajectories, with multiple values, or Points with one value per feature.

	    CALL DSG_PLTALONG_SETUP (dset, orientation, do_traj, 
     .			dsg_as_traj, dsg_as_time, status)
	    IF (status .NE. ferr_ok) GOTO 5000
	    IF (dsg_as_time) plot_axis(1) = t_dim
	    its_traj = do_traj

* Is this just a regular ribon plot with 3 variables (not a lon/lat traj plot)? 
* OR a PLOT/VS var1,var2 from a DSG dataset for var1,var2 not lon/lat?
* This may turn off ribbon_plot

	    IF (.NOT.dsg_as_traj .AND. .NOT.dsg_as_time) 
     .		CALL CHECK_PLOT_TRAJ (versus, numv, grid, its_traj)

	    IF (.NOT.its_traj) native_traj = .FALSE.
	    IF (its_traj) ribbon_plot = .TRUE.

*  numv = 1 if we're doing the trajectory ribbon plot automatically with PLOT var
*  Still allow doing it the old fashioned way numv = 2  for plot/vs lon, lat
*  If it's a trajectory dataset we still might send in say, lon,lat with a map projection
*  specified, so allow for that too.

	    IF ( (numv.EQ.1 .OR. nmv.EQ.3) .AND. its_traj) THEN 
	       ul_dolab(x_dim) = .FALSE.
	       ul_dolab(y_dim) = .FALSE.
	       ul_dolab(t_dim) = .TRUE.
	       ul_dolab(e_dim) = .TRUE.
	       ul_nlabs = 2

	       IF (orientation .EQ. pfeatureType_Point) THEN
		  use_line = .FALSE. 
		  IF (symbol .EQ. unspecified_int4) symbol = 20
	       ENDIF

	       IF (native_traj) THEN
		  its_traj = dsg_feature_type(dset) .EQ. pfeatureType_Trajectory ! as opposed to Point
		  CALL DSG_TRAJ_PLOT_SET_UP ( 
     .                overlay, no_labels, symbol, sym_size, color, color1, 
     .                use_line, do_dash, dashstyle, nokey, its_traj, only_val, 
     .                skipsym,  mv_list(1), cx_list(1), numv, indep_dat, dep_dat, status )

		  IF (numv.EQ.3 .AND. status.NE.ferr_ok) GOTO 2000  ! not traj after all, just a ribbon plot
	       
	       ELSE IF (do_traj .OR. dsg_as_traj) THEN
		  CALL  DSG_TRAJ_ALONGXY_SETUP( dset,
     .                 overlay, no_labels, symbol, sym_size, color, 
     .                 color1, use_line, do_dash, dashstyle, nokey,
     .                 only_val, skipsym, mv_list, cx_list, 
     .                 indep_dat, dep_dat, status )

	       ENDIF

	       GOTO 1000

	    ENDIF  ! its_traj

	     grid = igrd_save

	 ENDIF ! its_dsg

* Plot a variable e.g. as a set of timeseries per feature

* initialize

 2000	CONTINUE

	dep_max = arbitrary_small_val8
	dep_min = arbitrary_large_val8
	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	cx   = cx_list( 1 )
	mv   = mv_list( 1 )
	grid1 = cx_grid( cx )
	indep_dim = plot_axis(1)
	IF (dsg_as_time) THEN
	   indep_dim = t_dim
	   time_axis = .TRUE.
	   cal_id_1 = 1
	ENDIF
	nline_in_mem = 0
	phlim = qual_given( slash_hlimits )
	pvlim = qual_given( slash_vlimits )
	pxlim = qual_given( slash_xlimits )	! deprecated
	pylim = qual_given( slash_ylimits )	! deprecated
	cal_id_1 = 0
	cal_id_2 = 0
	cal_name = ' '
	dep_axtyp = 1
	indep_axtyp = 1
	mod_vs_x = .FALSE.
	mod_vs_y = .FALSE.
	axis_x_dir = no_dim
	axis_y_dir = no_dim
	adjust_time = .FALSE.
        cx_x = cx_list(1)
	xtra_lab = ' '
        nkey_entries = numv
	itmp = mnormal
	do_units = .TRUE.
	x_is_time = .FALSE.
	y_is_time = .FALSE.
	tstyle = ' '
	the_taxis = no_dim
	icode = 0
	ribbon_var = 0
	gap_var = 0
	t1_date = ' '
	tref = ' '

* set limit on the number of plot keys
        IF (.NOT.nokey  .AND.  numv.GT.max_key_entries) THEN
           buff1 = LEFINT( max_key_entries, slen  )
           CALL WARN('Too many lines to label. Labeling initial '
     .                          //buff1(:slen) )
           nkey_entries = max_key_entries
        ENDIF

* determine PLOT+ memory required
        plot_mem_used = 0
        DO ipl = 1, numv
	   gsize = CGRID_SIZE8( cx_list(ipl) )
	   IF (plot_mem_used + 2*step_inc*gsize .GT. 
     .         arbitrary_large_int4) GOTO 5300
	   plot_mem_used = plot_mem_used + 2*step_inc*gsize 
	ENDDO


* check for improper data supplied
	IF ( num_uvars_in_cmnd .LT. 2 ) GOTO 5100

* length of plot array
        gsize = CGRID_SIZE8(cx)
	IF (gsize .GT. arbitrary_large_int4) GOTO 5300
	npts = gsize	! (redundantly calc'd below) 5/90

* Overlay plots, keep the setting of IAUTOT that was set on the underlay plot.

	IF (overlay) THEN
	   iautot_save = iautot
	   iautot = 0
	ENDIF

* Mark the gap-variable for PLOT/VS 

	IF (grid_is_dsg .AND. .NOT.its_traj) addgaps = .TRUE.

	IF (addgaps) THEN
	   IF (ribbon_plot) THEN
	      gap_var = 4
	   ELSE
	      IF (num_uvars_in_cmnd .GT. 3) GOTO 5190
	      gap_var = 3
	   ENDIF
	ENDIF

* set flag indicating a 1D plot is on the screen
	IF (.NOT.overlay) onedee_on = .TRUE.

	IF (ribbon_plot) ribbon_var = 3

* decide if the picture should be rotated 90 degrees
	flip = transpz

	IF ( flip ) THEN
	   dep_ax    = 'X'
	   indep_ax  = 'Y'
	   dep_lab   = ppl_xlab
	   indep_lab = ppl_ylab
	   dep_len   = xlen
	   ind_len   = ylen
	   indep_is_log = is_logy
           dep_is_log   = is_logx

	   IF ( pxlim.GT.0 ) THEN
	      pindeplim = pxlim
	   ELSE
	      pindeplim = pvlim
	   ENDIF 
	   IF ( pylim.GT.0 ) THEN
	      pdeplim = pylim
	   ELSE
	      pdeplim = phlim
	   ENDIF 
	ELSE
	   dep_ax    = 'Y'
	   indep_ax  = 'X'
	   dep_lab   = ppl_ylab
	   indep_lab = ppl_xlab
	   ind_len   = xlen
	   dep_len   = ylen
	   indep_is_log = is_logx
           dep_is_log   = is_logy
	   IF ( pxlim.GT.0 ) THEN
	      pindeplim = pxlim
	   ELSE
	      pindeplim = phlim
	   ENDIF 
	   IF ( pylim.GT.0 ) THEN
	      pdeplim = pylim
	   ELSE
	      pdeplim = pvlim
	   ENDIF 
	ENDIF	   

* do not re-draw the main label, logo, other labels from the underlay plot
	IF (overlay) THEN
	   nlabs_on = 0  
	   CALL PPLCMD ( from, line, 0, 'LABS ', 1, 1 )	
	ENDIF

	IF ( no_labels ) CALL PPLCMD ( from, line, 0, 'LABS ', 1, 1 )

* * * * * * *    - - - INDEPENDENT AXIS DATA - - - * * * * * * * * *

	   time_axis =	.FALSE.
	   adjust_time = .TRUE.  ! (Says, do not try to further adjust time...)
	   CALL EXTRACT_LINE ( cx,
     .			    memry(mv)%ptr,
     .			    mv,
     .			    indep_dat,
     .			    indep_dim,
     .			    ndim,
     .			    npts,
     .			    status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   var1 = 2		! since first var is independent axis
	   ndv  = numv - 1	! # of dependent variables
	   use_keys = .FALSE.
	   plot_title = FULL_VAR_TITLE( cx_list(1), .TRUE., slen )

* ... Is this a dsg Profile dataset? Are we doing a vs plot? If so we may need to expand the 
*     first or second variable to the OBS axis.

	   expand_dsg_ok = .TRUE.
	   IF (its_dsg) THEN  ! Get the size of the second variable and decide whether to expand to obs axis.

	      gsize = CGRID_SIZE8( cx_list(var1) )
	      npts2 = gsize

	      expand_dsg_ok = (npts.NE.nftrsets .AND.
     .	          (npts.NE.nfeatures .OR. npts2.NE.nfeatures) )
	      dsg_fmt_grid = 0

	      IF (its_cmpnd) orientation = pfeatureType_Profile
	      IF ( orientation.EQ.pfeatureType_Profile .AND. 
     .	           versus .AND. expand_dsg_ok) THEN

	         IF (npts .EQ. nfeatures) CALL DSG_OBS_BY_FEATURE_VAR 
     .	             (dset, cx, nfeatures, obsdimlen, indep_dat)
	         npts = obsdimlen
	         dsg_fmt_grid = dsg_xlate_grid(dset)
	      ENDIF
	      IF (dsg_feature_type(dset).EQ.pfeatureType_Point) expand_dsg_ok = .TRUE.
	   ENDIF


* ... Is this a dsg dataset and we're doing a DSG plot (not a native trajectory plot)  then 
*     apply an obs mask. Resets npts.

	   IF (grid_is_dsg .AND. .NOT.its_traj .AND.expand_dsg_ok) 
     .            CALL PLOT_DSG_APPLY_OBS_MASK (dset, cx, nfeatures, obsdimlen, 
     .             indep_dat, npts)

* ... label the independent axis as the first var given

	      buff1 = VAR_UNITS(cx_x)
	      iunits = TM_UNIT_ID(buff1)

* Does it have time units? If so draw a formatted time axis
	      x_is_time = .FALSE.
	      since_T0 = MAX( INDEX(buff1,'since'), INDEX(buff1,'SINCE') )
	      slen = TM_LENSTR1(buff1)
	      IF (since_T0.GT.0 .AND. slen.GT.since_T0+5) THEN
	         cal_id_1 = 1
	         CALL TM_ALLO_DYN_LINE( itmp, status )	
	         CALL CD_GET_TIME_UNITS ( buff1, cal_id_1, line_units(itmp), 
     .			                   line_t0(itmp), dum, status )
		 IF (status .EQ. ferr_ok) THEN
		    x_is_time = .TRUE.
		    the_taxis = indep_dim
		    iunits = TM_UNIT_ID( line_units(itmp) )
		    line_unit_code(itmp) = TM_UNIT_ID( line_units(itmp) )
		    line_tunit(itmp) = un_convert(line_unit_code(itmp))
		    line_cal_name(itmp) = 'GREGORIAN'
		    line_direction( itmp ) = 'TI'

		    CALL MINMAX( indep_dat, npts, mr_bad_data(mv), lo, hi, nok )
		    grid_line(the_taxis, mgrid_buff) = itmp
		 
		    igrd_save =  cx_grid(cx_list(1))
		    cx_grid(cx_list(1)) = mgrid_buff
		    ww_save = cx_lo_ww(the_taxis,cx_list(1) ) 
		    cx_lo_ww(the_taxis,cx_list(1)) = lo

		    ind_min = TSTEP_TO_SECS (mgrid_buff, the_taxis, lo)
		    ind_max = TSTEP_TO_SECS (mgrid_buff, the_taxis, hi)

		    IF (dsg_fmt_grid .GT. 0) THEN
		       ind_min = TSTEP_TO_SECS (dsg_fmt_grid, t_dim, lo)
		       ind_max = TSTEP_TO_SECS (dsg_fmt_grid, t_dim, hi)
		    ENDIF

* Plotting short time range? If so recompute tref     
* axis length in hours from time since BC in seconds
		    IF (.NOT.overlay) THEN
                       hrs = ( ind_max - ind_min ) / 3600.
                       IF (hrs.LT.24) tref = 
     .                  MERGED_WHOI_DATE( cx_list(var1), the_taxis, numv, .TRUE. )

                       all_1_ind = ind_min .EQ. ind_max
                       IF ( all_1_ind ) THEN
                          ind_min = ind_min - 1.
                          ind_max = ind_max + 1.
                       ENDIF
		    
                       CALL TAXIS_STYLE( indep_ax, ind_min, ind_max, tstyle, xtra_lab )
* 11/16 Get the min and max as used by PPLUS.  That function uses WHOI formatted time
                      CALL TPLOT_AXIS_ENDS (ind_min, ind_max, cal_id_1, tstyle)
 
                      IF (.NOT.overlay) CALL AXIS_END_SYMS( indep_ax,
     .                          SECS_TO_TSTEP( mgrid_buff, the_taxis, ind_min ),
     .                          SECS_TO_TSTEP( mgrid_buff, the_taxis, ind_max ) )

		    ENDIF

	         ELSE
		    IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
		    itmp = mnormal
	         ENDIF
	      ELSE  ! not time axis
	      
	         CALL MINMAX( indep_dat, npts, mr_bad_data(mv), lo, hi, nok )
	         ind_min = MIN(ind_min, lo)
	         ind_max = MAX(ind_max, hi)
	      ENDIF

	      slen = TM_LENSTR1(plot_title)

*  Check for lon-360 issues? Crossing the dateline within features, 
*  when longitude is in the obs direction.

              CALL GEOG_LABEL_VS (buff1, iunits, x_dim, axis_x_dir)

              IF (iunits .EQ. 4) THEN
		 delta = MAX(1., (ind_max-ind_min)/10.)  ! just a rough guess, will be set later
		 CALL GET_AXIS_FORMAT( ind_min, ind_max, delta, fmt, use_nice )

		 IF (grid_is_dsg .AND. versus .AND. use_nice .AND.
     .               axis_x_dir.EQ.x_dim .AND. npts.NE.nfeatures) THEN
		    
		    user_hlim = pindeplim .GT. 0
		    CALL PLOT_DSG_CHECK_LON360 (dset, cx, indep_dat, mr_bad_data(mv), 
     .                    npts, nfeatures, dum1, dum2, .TRUE.)

                 ENDIF
              ENDIF

	      IF (.NOT.no_labels .AND. iunits .NE. 4) 
     .           CALL BOX_LABEL( indep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*ind_len, 
     .                     dflt_letsize_label*textscale,
     .			   dflt_letsize_label*textscale, 
     .                     ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )


* ... force axis scaling if the data has no range
	   all_1_ind = NO_LINE_RANGE( indep_dat,npts,mr_bad_data(mv),val1 )
	   delta = unspecified_val8

	   IF ( pindeplim .GT. 0 ) THEN
	      IF ( pxlim.GT.0 .AND. .NOT.denig_xylim_msg_done ) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
* Get calendar name for equal_range (even if not time axis)
	      tax = grid_line(f_dim,cx_grid(cx_list(var1)) )
              IF (tax .EQ. 0) tax = grid_line(t_dim,cx_grid(cx_list(var1)) )
              cal_name = line_cal_name(tax)
              cal_id_1 = TM_GET_CALENDAR_ID ( cal_name )

	      CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pindeplim):qual_end(pindeplim)),
     .			indep_dim, ind_min, ind_max, delta,
     .			formatted, range_rqd, cal_id_1, status )

* Check for a valid axis if they've asked for a log axis before proceeding

	      IF (indep_is_log) THEN
	         CALL AXIS_ENDS(indep_ax,indep_dim,grid1,ind_min,ind_max,
     .                       delta, indep_is_log, indep_axtyp, versus, status)
	         IF ( status .NE. ferr_OK ) THEN
                    first = ind_min
                    last  = ind_max
                    GOTO 5170
                 ENDIF

* Get requested ends in data units again
	         CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pindeplim):qual_end(pindeplim)),
     .			indep_dim, ind_min, ind_max, delta,
     .			formatted, range_rqd, cal_id_1, status )

	      ENDIF

	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ELSEIF ( all_1_ind .AND. .NOT.overlay ) THEN
	      IF ( val1 .EQ. mr_bad_data(mv) ) THEN
		 delta = 1.
	         ind_min = 0.D0
	         ind_max = 1.D0
	      ELSE
	         delta = 0.1* ABS(val1)
	         ind_min = val1 - delta
	         ind_max = val1 + delta
	      ENDIF
	   ELSE
	      CALL MINMAX( indep_dat, npts, mr_bad_data(mv), lo, hi, nok )
	      IF (nok .EQ. 0) THEN ! all missing
	         ind_min = -1.
	         ind_max =  1.
	      ElSE
	         ind_min = lo
	         ind_max = hi
	      ENDIF
	   ENDIF

	   IF (.NOT. all_1_ind .AND. .NOT.indep_is_log) THEN
	      pad = (ind_max - ind_min)* 0.01
	      ind_min = ind_min - pad
	      ind_max = ind_max + pad
	   ENDIF

* Set up modulo longitude handling if /vs and variable has units
* of longitude

	   cx_x = cx_list(1)	 
	   cx_y = cx_list(2)
	   buff1 = VAR_UNITS(cx_x)
	   IF (iunits .GE. 0) iunits = TM_UNIT_ID(buff1)

	   IF ( .NOT.overlay .OR.  set_up_axes ) THEN

	      IF ( .NOT.overlay ) CALL AXIS_ENDS(indep_ax,indep_dim,grid1,ind_min,ind_max, 
     .                       delta, indep_is_log, indep_axtyp, versus, status)

	      IF ( status .NE. ferr_ok ) THEN
                 first = ind_min
                 last  = ind_max
                 GOTO 5170
              ENDIF

* This routine checks the units and the setting for formatted lon/lat (time?) axes
* If the formatting has been turned off, resets iunits and flag mod_vs_x.
* If the axes are labeled with longitude, latitude, or time, then set ul_dolab to not
* put upper-left labels on the plot with the ranges.

              CALL GEOG_LABEL_VS (buff1, iunits, x_dim, axis_x_dir)

              IF (iunits .EQ. 4) THEN
		 IF (delta .EQ. unspecified_val8) delta = MAX(1., (ind_max-ind_min)/10.)  ! case of plot/over/axes; delta hasn't been set
		 CALL GET_AXIS_FORMAT( ind_min, ind_max, delta, 
     .                  fmt, use_nice )
                 IF (use_nice) THEN
                    plot_axis(1) = 1
		    IF (axis_x_dir .EQ. x_dim) THEN
                       ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                  ',''''LONE'''')'
                       CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
                       CALL LON_LAT_FMT (x_dim, 'X')
                       mod_vs_x = .TRUE.
                       ul_dolab(x_dim) = .FALSE.

                    ELSEIF (axis_x_dir .EQ. y_dim) THEN
                       ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                  ',''''LAT'''')'
                       CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
                       CALL LON_LAT_FMT (y_dim, 'Y')
                       ul_dolab(y_dim) = .FALSE.
		       plot_axis(1) = 2
                    ELSE
                       ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                  ',''''LONE'''')' 
                       CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
                       CALL LON_LAT_FMT (x_dim, 'X')
                       ul_dolab(x_dim) = .FALSE.
                    ENDIF
                 ENDIF
              ELSEIF (iunits .LT. 0) THEN ! time axis formatting
                 CALL PPLCMD ( from, line, 0, 'TIME', 1, 1 )
                 CALL PPLCMD ( from, line, 0, 'TAXIS 60,ON', 1, 1 )
                 ppl_buff = ' '
                 WRITE ( ppl_buff, 3004 ) 'GREGORIAN'
                 CALL PPLCMD (from, line, 0, ppl_buff, 1, 1)
                 ul_dolab(t_dim) = .FALSE.
		 plot_axis(1) = 4
	      ELSE
                 CALL PPLCMD ( from, line, 0, 'XFOR 0', 1, 1 )	 
              ENDIF	
		  
              CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 ) ! for now, anyway	
	      
	   ELSE ! overlay
              CALL PPLCMD ( from, line, 0, indep_ax//'FOR 0', 1, 1 )

              IF (iunits .EQ. 4) THEN
	         mod_vs_x = (TM_HAS_STRING(buff1, '_e') .OR.
     .                   TM_HAS_STRING(buff1, 'lon') )
	         
              ENDIF

              buff1 = VAR_UNITS(cx_y)	 
              junits = TM_UNIT_ID(buff1)
              IF (junits .EQ. 4) THEN
                  mod_vs_y = (TM_HAS_STRING(buff1, '_e') .OR.
     .                        TM_HAS_STRING(buff1, 'lon') ) 
	   ENDIF

	ENDIF

* set up PLOT5 to ignore bad data flag
	ppl_buff = ' '
	WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), indep_ax
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1)


 3004   FORMAT ('CALENDAR ', A7)
 3005	FORMAT ( 'LIMITS ',G15.8,1X,A1,'EQ' )

* Send info about gap var for PLOT/VS/GAPLOC

	CALL PPLCMD ( from, line, 0, 'GAPLOC 0', 1, 1)
	IF (addgaps) THEN
	   ppl_buff = ' '
	   WRITE ( ppl_buff, 3007 ) gap_var
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1)
	ENDIF
 3007	FORMAT ( 'GAPLOC ',I5 )

* * * * * * *    - - - LOAD DATA - - - * * * * * * * * *
* 11/95: determine T0 reference time for time axis
	IF ( time_axis .AND. .NOT.overlay ) THEN
	   tref = MERGED_WHOI_DATE( cx_list(var1), the_taxis, numv, .FALSE. )
	ELSE
	   tref = ' '		! overlays reuse previous tref
	ENDIF

* loop through all the dependent variables loading them into PPLUS

	all_1_dep = .TRUE.
	set_axis = .TRUE.
	only_val  = bad_val4
	nload_all = 0

	IF (grid_is_dsg .AND. .NOT.its_traj .AND. use_line ) numv = numv + 1 ! fake a gap-var

	DO 200 ipl = var1, numv
	   IF (grid_is_dsg .AND. .NOT.its_traj .AND. ipl.EQ.gap_var) THEN

*  create the gaps-var.  IF npts2 = nfeatures (e.g. a plot of lon vs lat for timeseries stations)
*  then the gap-var will be all 0's, e.g. no gaps.
	      cx = cx_list( 1 )
	      IF (npts2 .NE. nfeatures) npts2 = obsdimlen
	      CALL DSG_OBS_MARK_GAPS (dset, cx, nfeatures, npts2, dep_dat)
	      status = ferr_ok
	   
	   ELSE
	      mv   = mv_list( ipl )
	      cx   = cx_list( ipl )
	      grid = cx_grid( cx )
	      dset = cx_data_set(cx)

* set up a dependent variable.

              IF (dsg_as_time) THEN
	         mv   = mv_list( 1 )
	         cx = cx_list(1)
		 cx_lo_ss(cx, z_dim) = cx_lo_ss(cx_list(ipl+1), t_dim)
		 cx_hi_ss(cx, z_dim) = cx_hi_ss(cx_list(ipl+1), t_dim)
		 IF (cx_hi_ss(cx, z_dim) .EQ. unspecified_int4) EXIT  ! loop 200
              ENDIF
	      CALL EXTRACT_LINE (	cx,
     .				memry(mv)%ptr,
     .				mv,
     .				dep_dat,
     .				dep_dim,
     .				ndim,
     .				npts2,
     .				status	)
           ENDIF
	   iaxis = CGRID_AXIS(dep_dim,cx)
           IF (dsg_as_time) THEN
	      dep_dim = t_dim
	      cx = cx_list( ipl ) + 1  ! for indep-data, each line
	      mv   = mv_list( 1 )
	   ENDIF
 
	   IF ( status .NE. ferr_ok ) GOTO 5000

	   IF (cxdsg_empty_set) THEN
	      dep_dat(1) = -1.
	      dep_dat(2) =  1.
	   ENDIF
	      

* ... If this is a profile or timeseries collection maybe we are just making a map with 
*     the stations so dep_dat and indep dat are the longitudes and latitudes. The flag
*     expand_dsg_ok false, says just plot the instance variables
	   
* ... Or, is this a dsg Profile dataset? Are we doing a vs plot? If so we may need to expand the 
*     first or second variable to the OBS axis.

	   IF (its_dsg .AND. 
     .	       orientation.EQ.pfeatureType_Profile .AND. 
     .	       npts2.EQ.nfeatures .AND. expand_dsg_ok) THEN
	      IF (npts2 .EQ. nfeatures) CALL DSG_OBS_BY_FEATURE_VAR 
     .	           (dset, cx, nfeatures, obsdimlen, dep_dat)
	      npts2 = obsdimlen
	      dsg_fmt_grid = dsg_xlate_grid(dset)
	   ENDIF

* ... For a dsg dataset, can ask to color by a feature-level variable
*     such as an ID or a string-valued feature variable. 

	   doing_ribbon_color = ribbon_plot .AND. ipl.EQ.ribbon_var

	   IF (its_dsg .AND. doing_ribbon_color) THEN

	      color_dsg_id = (cx_variable(cx).EQ.dsg_feature_var(dset) )  .OR. 
     .		             (npts2.LT.obsdimlen .AND. cx_type(cx).EQ.ptype_string)
	      IF (color_dsg_id ) CALL SET_DSG_ID_LEV
     .		      (dset, cx, nfeatures, dep_dat, mr_type(mv), changed_key)

	      IF (npts2 .EQ. nfeatures) THEN
		 IF (color_dsg_id) THEN
		    DO i = 1, npts2
		       dep_dat(i) = i
		    ENDDO
		 ENDIF
		 IF (expand_dsg_ok) THEN
		    CALL DSG_OBS_BY_FEATURE_VAR (dset, cx, nfeatures, obsdimlen, dep_dat)
		    npts2 = obsdimlen
	         ENDIF
	      ENDIF

	   ENDIF

* ... Is this a dsg dataset?  if so apply an obs mask. Resets npts2

	   IF (grid_is_dsg .AND. .NOT.its_traj .AND. 
     .		expand_dsg_ok .AND. .NOT.cxdsg_empty_set) 
     .       CALL PLOT_DSG_APPLY_OBS_MASK (dset, cx, nfeatures, 
     .       obsdimlen, dep_dat, npts2)

* ... replicate points to create a "step" plot if requested  (1/01)
	   nload = npts2 * step_inc
           IF (step_inc .EQ. 2) THEN
	      DO i = npts2, 1, -1
	         dep_dat(2*i)   = dep_dat(i)
	         dep_dat(2*i-1) = dep_dat(i)
	      END DO
           ELSEIF (step_inc .EQ. 3) THEN
	      DO i = npts2, 1, -1
	         dep_dat(3*i)   = mr_bad_data(mv)
	         dep_dat(3*i-1) = dep_dat(i)
	         dep_dat(3*i-2) = dep_dat(i)
	      END DO
	   ENDIF

* set up corresponding independent axis
* ( note - each variable may have different points on the indep. axis )

* ... number of values must be equal in independ. and depend. for ordered pairs
	      IF ( npts2 .NE. npts ) GOTO 5130

* Set up the dependent axis.

	      set_axis = .TRUE.
	      IF (ribbon_plot .OR. addgaps) set_axis = (ipl .EQ. var1)
              IF (set_axis) THEN
	         IF (.NOT.overlay) THEN
	            CALL MINMAX( dep_dat, npts, mr_bad_data(mv), lo, hi, nok )
 	            IF (nok .EQ. 0) THEN ! all missing
 	               dep_min = -1
 	               dep_max =  1
 	            ELSE
 	               dep_min = MIN(dep_min, lo)
 	               dep_max = MAX(dep_max, hi)
 	            ENDIF
		 ENDIF
		 
		 all_1_dep = NO_LINE_RANGE( dep_dat, nload, mr_bad_data(mv), val1 )
		 
		 IF (.NOT.all_1_dep .AND. .NOT.dep_is_log) THEN
		    pad = (dep_max - dep_min)* 0.01
		    dep_min = dep_min - pad
		    dep_max = dep_max + pad
		 ENDIF

                 cx_y = cx_list(2) 	
                 buff1 = VAR_UNITS(cx_y)
		 IF (doing_ribbon_color) buff1 = ' '
		 
		 iunits = TM_UNIT_ID(buff1)

c	         IF ( .NOT.overlay .OR.  set_up_axes ) THEN
	         IF ( .NOT.overlay  ) THEN
* Does it have time units? If so draw a formatted time axis
                 y_is_time = .FALSE.
                 since_T0 = MAX( INDEX(buff1,'since'), INDEX(buff1,'SINCE') )
                 slen = TM_LENSTR1(buff1)
                 IF (.NOT.x_is_time .AND. since_T0.GT.0 .AND. slen.GT.since_T0+5) THEN
	            cal_id_1 = 1
	            CALL TM_ALLO_DYN_LINE( itmp, status )	
	            CALL CD_GET_TIME_UNITS ( buff1, cal_id_1, line_units(itmp), 
     .			                   line_t0(itmp), dum, status )
		    IF (status .EQ. ferr_ok) THEN
		       y_is_time = .TRUE.
		       time_axis = .TRUE.
		       the_taxis = dep_dim
		       has_time_axis = .TRUE.
		       iunits = TM_UNIT_ID( line_units(itmp) )
		       line_unit_code(itmp) = TM_UNIT_ID( line_units(itmp) )
		       line_tunit(itmp) = un_convert(line_unit_code(itmp))
		       line_cal_name(itmp) = 'GREGORIAN'
		       line_direction( itmp ) = 'TI'

		       grid_line(the_taxis, mgrid_buff) = itmp
		 
		       igrd_save =  cx_grid(cx_list(1))
		       cx_grid(cx_list(1)) = mgrid_buff
		       ww_save = cx_lo_ww(the_taxis,cx_list(1) ) 
		       cx_lo_ww(the_taxis,cx_list(1)) = lo

                       dep_min = TSTEP_TO_SECS (mgrid_buff, the_taxis, lo)
                       dep_max = TSTEP_TO_SECS (mgrid_buff, the_taxis, hi)

		       IF (dsg_fmt_grid .GT. 0) THEN
		          dep_min = TSTEP_TO_SECS (dsg_fmt_grid, t_dim, lo)
		          dep_max = TSTEP_TO_SECS (dsg_fmt_grid, t_dim, hi)
		       ENDIF

		       CALL TAXIS_STYLE( dep_ax, dep_min, dep_max, tstyle, xtra_lab )

* 11/16 Get the min and max as used by PPLUS.  That function uses WHOI formatted time
		       CALL TPLOT_AXIS_ENDS (dep_min, dep_max, cal_id_1, tstyle)
 
		       CALL AXIS_END_SYMS( dep_ax,
     .                          SECS_TO_TSTEP( mgrid_buff, the_taxis, dep_min ),
     .                          SECS_TO_TSTEP( mgrid_buff, the_taxis, dep_max ) )

		    ENDIF

		 ENDIF

* This routine checks the units and the setting for formatted lon/lat (time?) axes
* If the formatting has been turned off, resets iunits and flag mod_vs_x.

		 CALL GEOG_LABEL_VS (buff1, iunits, y_dim, axis_y_dir)
                 IF (iunits .EQ. 4) THEN
	            delta = unspecified_val8
		    
	            CALL AXIS_ENDS(dep_ax,dep_dim,grid1,dep_min,dep_max,
     .                      delta, dep_is_log, dep_axtyp, versus, status)
	            CALL GET_AXIS_FORMAT( dep_min, dep_max, delta, 
     .                 fmt, use_nice )

                    IF (use_nice) THEN

* Make sure if we are using lon/lat labeling that we have just one of each direction.
* e.g. if they said just degrees_n for the x dir, but just "degrees" for the y var, 
* use east for the vert. If they said degrees_e for the y var but just degrees for the x,
* use lon for the y axis.
* If the axes are labeled with longitude, latitude, or time, then set ul_dolab to not
* put upper-left labels on the plot with the ranges.

                       IF (axis_x_dir.EQ.y_dim .AND. axis_y_dir.EQ.no_dim) THEN
		          IF (iunits .EQ. 4) THEN
		             ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                       ',''''LONE'''')' 
		             ul_dolab(x_dim) = .FALSE.
		             plot_axis(1) = 1
		          ENDIF

                       ELSEIF (axis_y_dir .EQ. x_dim) THEN
		          ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                       ',''''LONE'''')' 
		          mod_vs_y = .TRUE.
		          ul_dolab(x_dim) = .FALSE.
		          plot_axis(1) = 1

		          IF (axis_x_dir .EQ. no_dim) THEN
		             buff1 = VAR_UNITS(cx_x)
		             iiunits = TM_UNIT_ID(buff1)
		             CALL GEOG_LABEL_VS (buff1, iiunits, x_dim, axis_x_dir)
		             IF (iiunits .EQ. 4) THEN
		                ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                           ',''''LAT'''')'
		                ul_dolab(y_dim) = .FALSE.
		                plot_axis(1) = 2
		             ENDIF
		          ENDIF
                       ELSEIF (axis_y_dir .EQ. y_dim) THEN
                           ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                       ',''''LAT'''')'
                           ul_dolab(y_dim) = .FALSE.
		           plot_axis(2) = 2
                       ELSE
                           ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                       ',''''LAT'''')' 
                           ul_dolab(y_dim) = .FALSE.
		           plot_axis(2) = 2
                       ENDIF

                       CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
                    ENDIF

                 ELSEIF (iunits .LT. 0) THEN ! time axis formatting
                    CALL PPLCMD ( from, line, 0, 'TIME', 1, 1 )
                    CALL PPLCMD ( from, line, 0, 'TAXIS/YAXIS 60,ON', 1, 1 )
                    ppl_buff = ' '
                    WRITE ( ppl_buff, 3004 ) 'GREGORIAN'
                    CALL PPLCMD (from, line, 0, ppl_buff, 1, 1)
                    ul_dolab(t_dim) = .FALSE.
		    plot_axis(2) = 4

	         ELSE
                   CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 )	 
                 ENDIF	
	         ENDIF  ! not overlay
              ENDIF  ! set_axis

* Compute the mean and standard dev for ribbon-color variable, 
* needed for computing color levels. Results stored in PPLUS common.
* If histogram-based levels are requested, compute the 
* histogram bins.
* send REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* If the user set some /LEVELS then compute_mnstd just returns 

	   IF (doing_ribbon_color) THEN

              IF (need_histo) THEN
* create temporary buffer to contain workspace
                 CALL CREATE_TEMP_MEM_VAR( cx, mvh_temp, status )
                 IF ( status .NE. ferr_ok ) RETURN
                 plot_mem_used = plot_mem_used + npts
                 CALL COMPUTE_HISTO_BINS (dep_dat, memry(mvh_temp)%ptr, 
     .            mr_bad_data(mv), npts, status) 
     
* ... clean up temporary variable
                 CALL DELETE_VARIABLE( mvh_temp )
              ELSE 
		 rbad = bad_val4
                 CALL COMPUTE_MNSTD (dep_dat, mr_bad_data(mv), need_std, npts, rbad, status)
                 IF (need_std .AND. status.NE.ferr_ok) THEN
* ... set up for automatic levels
                    CALL PPLCMD ( from, line, 0, 'LEV,()', 1, 1 )
                    CALL USE_LINEAR_LEVELS
                    status = ferr_ok
                 ENDIF

              ENDIF

	   ENDIF

* ... check that there is a range of dependent data for PLOT+ auto-scaling
* ... when all the variables are considered together (but not the ribbon var)

	   IF (addgaps .AND. ipl.EQ.gap_var) GOTO 199

	   IF (.NOT.ribbon_plot .OR. (ribbon_plot .AND. ipl.LT.ribbon_var)) THEN
	      this_no_range(ipl) = .FALSE.
              IF (set_axis) this_no_range(ipl) = NO_LINE_RANGE( 
     .          dep_dat, nload, mr_bad_data(mv), val1 )
	      all_1_dep = all_1_dep .AND. this_no_range(ipl)
              IF (this_no_range(ipl) .AND. val1.NE.mr_bad_data(mv) )
     .           this_no_range(ipl) = .FALSE. ! Keep the value for putting NO VALID on labels

	      IF ( all_1_dep ) THEN
	         IF ( only_val .EQ. bad_val4 ) THEN
	            IF ( val1 .NE. mr_bad_data(mv) ) only_val = val1
	         ELSE
	           all_1_dep = all_1_dep .AND. only_val .EQ. val1
	         ENDIF

	      ENDIF
	   ENDIF  ! checking range when not ribbon variable

  199	   CONTINUE  ! checking range when not gap variable

* pass the data to PLOT+

	   WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), dep_ax
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   IF ( time_axis ) THEN
              IF ( tax .EQ. mnormal .OR. tax .EQ. munknown ) THEN  ! 8/91 bug
                 dt_min = 1.0
              ELSE
	         dt_min = line_tunit( grid_line(the_taxis,grid) ) / 60.! sec-->min
	         IF ( ITSA_TRUEMONTH_AXIS(grid_line(the_taxis,grid)) ) 
     .                dt_min = un_convert(pun_day)/60. 

*  Check for calendar mismatch for time- overlay plots
	         IF (overlay) THEN
                    IF (cal_id_1 .NE. saved_calendar_id) THEN
                       cal_id_old = saved_calendar_id
                       cal_id_new = cal_id_1
                       GO TO 5160
	            ENDIF
                 ENDIF
                 t1_date = INTERNAL_WHOI_DATE( grid, the_taxis, 1.0D0 )  ! moved up one line...
              ENDIF
           ELSE
	      IF (overlay .AND. has_time_axis) THEN
	         dt_min  = saved_dt_min
	         t1_date = saved_t1_date
	      ELSEIF (.NOT.ribbon_plot) THEN
                 dt_min  = 1.0
                 t1_date = ' '
	      ENDIF
           ENDIF

	   IF (overlay) THEN
	      SOVER = .true.
	   ELSE
	      has_time_axis = time_axis
 	      IF (has_time_axis) THEN
	         saved_dt_min  = dt_min		! to re-use on PLOT/VS or POLYGON
	         saved_t1_date = t1_date
	         saved_calendar_id = cal_id_1
	      ENDIF
	   ENDIF

* put colorvar in the right place
	   IF (doing_ribbon_color) THEN   !!! .AND. .NOT.add_ribbon
	      flip = .FALSE.
	      IF (indep_dim .EQ. z_dim) THEN
	         WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), indep_ax
	         CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	      ENDIF
	   ENDIF

! put gap-var in the right place
	   IF (ipl .EQ. gap_var) THEN
	      flip = .FALSE.
	      IF (indep_dim .EQ. z_dim) THEN
	         WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), indep_ax
	         CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	      ENDIF
	   ENDIF

* If a /vs plot has a variable with units of time, put the itmp line into 
* the scratch grid mgrid_buff, so the WHOI_DATE functions can get at the 
* line_* information stored above after calls to CD_GET_TIME_UNITS. 

	      IF (x_is_time) THEN
		 dt_min = line_tunit(itmp)/ 60.
		 
		 adjust_time = .FALSE.
		 CALL AXIS_ENDS(indep_ax,the_taxis,mgrid_buff,ind_min,ind_max,
     .                       delta, indep_is_log, indep_axtyp, versus, status)

                 tref = MERGED_WHOI_DATE( cx_list(1), the_taxis, 1, .FALSE. )
                 t1_date = INTERNAL_WHOI_DATE( mgrid_buff, the_taxis, 1.0D0 )

		 cx_grid(cx_list(1)) = igrd_save
		 cx_lo_ww(the_taxis,cx_list(1) ) = ww_save 
		 IF (ipl .EQ. numv) THEN
		    IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
		    itmp = mnormal
		 ENDIF
	      
	      ENDIF

	      IF ( y_is_time .AND. .NOT.doing_ribbon_color) THEN
	      
		 dt_min = line_tunit(itmp)/ 60.
		 
		 adjust_time = .FALSE.
		 CALL AXIS_ENDS(dep_ax,the_taxis,mgrid_buff,dep_min,dep_max,
     .                       delta, indep_is_log, indep_axtyp, versus, status)

                 tref = MERGED_WHOI_DATE( cx_list(1), the_taxis, 1, .FALSE. )
                 t1_date = INTERNAL_WHOI_DATE( mgrid_buff, the_taxis, 1.0D0 )

		 cx_grid(cx_list(1)) = igrd_save
		 cx_lo_ww(the_taxis,cx_list(1) ) = ww_save 
		 IF (ipl .EQ. numv) THEN
		    IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
		    itmp = mnormal
		 ENDIF


		 IF (.NOT.overlay) THEN
	            saved_dt_min  = dt_min		! to re-use on PLOT/VS or POLYGON
	            saved_t1_date = t1_date
	            saved_calendar_id = cal_id_1
	         ENDIF

		 IF (iunits .LT. 0) THEN
	            dt_min = saved_dt_min
	            t1_date = saved_t1_date
	            cal_id_1 = saved_calendar_id
		 ENDIF 
		         
	      ENDIF

* See pplldx comments. When loading the color-by variable lon vertical time axis
* this forces the right combination of settings.

	   IF (doing_ribbon_color) icode = 2
	   IF (y_is_time .AND. doing_ribbon_color) icode = 7

* Did we rescale a time axis in axis_format?  If so need to change time units.
* dt_min only matters if we're overlaying on a time axis

	   IF (iunits.GE.0 .AND. rescale_time) dt_min = 1./60.

	   IF ( flip ) THEN
	      CALL PPLLDX_envelope(icode,dep_dat,indep_dat,nload,
     .                             t1_date, tref, dt_min, plot_mem_used)
	   ELSE
	      CALL PPLLDX_envelope(icode,indep_dat,dep_dat,nload,
     .                             t1_date, tref, dt_min, plot_mem_used)
	   ENDIF
* ... increment number of lines on plot
	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1
* ... assign line style for the data plotting:
	   CALL LINE_STYLE(line_symbol, sym_size, skipsym, color, color1, use_line,
     .			   do_dash, dashstyle, nline_in_mem, nline_on)

	   nload_all = MAX(nload, nload_all)
 200	CONTINUE

        nline_on = nline_on + color1 - 1

* axis scaling and formatting
* ... independent axis

	IF ( .NOT.overlay ) THEN

* ... dependent axis scaling
*  force axis scaling if the data has no range

           IF (the_taxis .EQ. 0) the_taxis = dep_dim

	   IF ( pdeplim .GT. 0 ) THEN
	      IF (pylim.GT.0 .AND. .NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
*  Check for valid log axis before proceeding
*  First make sure the min and max are set.
	      IF (dep_is_log) THEN
	      
	         CALL MINMAX( dep_dat, npts, mr_bad_data(mv), lo, hi, nok )
		 IF (nok .EQ. 0) GOTO 5171
	         dep_min = lo
	         dep_max = hi
	         
	         CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, dep_is_log, dep_axtyp, versus, status )

	      ENDIF

* Do the dependent variable(s) have units of time? If not then don't send 
* dep_dim = t_dim to EQUAL_RANGE.  When the dependent data is on a time 
* axis, then the_taxis is set, but we don't want to be translating limits 
* for the dependent axis as time units.

	      dep_dir = dep_dim
	      IF (dep_dim .EQ. t_dim) THEN

	         DO ipl = 1, nmv
		    buff1 = VAR_UNITS( cx_list(ipl) )
		    iunits = TM_UNIT_ID(buff1)
		    IF ( iunits .GT. pun_last_time ) dep_dir = 0
		 ENDDO
		 
	      ENDIF

	      CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pdeplim):qual_end(pdeplim)),
     .			dep_dir, dep_min, dep_max, delta,
     .			formatted, range_rqd, cal_id_1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000

	      CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, dep_is_log, dep_axtyp, versus, status )


	   ELSEIF ( all_1_dep ) THEN
	      IF (only_val .EQ. bad_val4) val1 = 0.0	! 10/99
	      delta = 1.
	      IF (val1 .NE. 0) delta = 0.1* ABS(val1)
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid,val1-delta,val1+delta,delta,
     .                        dep_is_log, dep_axtyp, versus, status )   

	   ELSEIF (dep_is_log) THEN	! get dependent var scaling and set up log axis
	      CALL MINMAX( dep_dat, npts, mr_bad_data(mv), lo, hi, nok )
	      IF (nok .EQ. 0) GOTO 5171  ! all missing
	      dep_min = lo
	      dep_max = hi
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, dep_is_log, dep_axtyp, versus, status )

	   ELSE 
       
	      IF (its_dsg .AND. versus) THEN  ! /vs plot with depth as an axis?
	         dsg_fmt_grid = dsg_xlate_grid(dset)
	         cx = cx_list( 2 )
	         iaxis = grid_line(z_dim,dsg_fmt_grid)
	         IF (cx_variable(cx) .EQ. dsg_coord_var(z_dim,dset) .AND. 
     .       	     line_direction(iaxis) .EQ. "UD") THEN
	            tmp = dep_min
	            dep_min = dep_max
	            dep_max = tmp
	         ENDIF

	         delta = unspecified_val8
	         CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, dep_is_log, dep_axtyp, versus, status )
	      ENDIF

	   ENDIF
	   IF ( status .NE. ferr_OK ) THEN
	      first = dep_min
	      last  = dep_max
	      GOTO 5170
	   ENDIF

*   When not an overlay, set the axis type to log or reverse log

	   IF (dep_is_log  .OR.  indep_is_log) THEN
	      IF (flip) THEN
		 WRITE (val_buff, 3006) dep_axtyp, indep_axtyp
	      ELSE
 	         WRITE (val_buff, 3006) indep_axtyp, dep_axtyp
	      ENDIF

 3006         FORMAT ('axtype,', I2, ',', I2)
              CALL PPLCMD ( from, line, 0, val_buff, 1, 1)
	   ENDIF

	   IF (cxdsg_empty_set) THEN
	      WRITE ( val_buff, '(3(E14.7,1X))' ) dep_dat(1), dep_dat(2), dep_dat(2)
	      CALL PPLCMD ( from, line, 0, 
     .				dep_ax//'AXIS '//val_buff , 1, 1 )
              WRITE (cxdsg_axlabp_save, "(I2, ',', I2)") LABELX, LABELY
	      IF (flip) THEN
	         CALL PPLCMD (from, line, 0,'AXLABP,0,-1',1,1)
              ELSE
	         CALL PPLCMD (from, line, 0,'AXLABP,-1,0',1,1)
	      ENDIF
	   ENDIF

           ax1 = axdir(indep_dim)
  	   IF (flip) THEN
              CALL PPLCMD ( from, line, 0, 'SET AX_VERT '//ax1, 1, 1 )
	   ELSE
	      CALL PPLCMD ( from, line, 0, 'SET AX_HORIZ '//ax1, 1, 1 )
           ENDIf
	ENDIF

*  TITLES
* Main plot title. Add labels for multi-lines as a key, 
* or a "Colored by var2" label for the ribbon color variable
*
	hlen = xlen ! single-precision xlen from PPLUS common -> double prec. var

        IF (x_is_time .OR. y_is_time) THEN
           ul_dolab(t_dim) = .FALSE.
           READ (t1_date(13:14), *) cen
           READ (t1_date(1:2), *) yr
           ind_min = 100*cen + yr 
        ENDIF

* Line-key labels

	IF ( .NOT.no_labels ) THEN
	   
	   CALL LINE_PLOT_LABELS (var1, nkey_entries, ndv, cx_list, 
     .		this_no_range, overlay, versus, nokey, time_axis, 
     .		tstyle, cal_id_1, ribbon_var, indep_lab, dep_lab, 
     .		ind_min, dep_len, hlen, 0, dset, 0)
	ENDIF

* successful completion
 1000	CONTINUE
	status = ferr_ok
	IF (overlay) iautot = iautot_save
	
	IF (dsg_as_time) nmv = nmv + 1  ! for un-doing all the extra contexts

	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
 	IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
	IF (dsg_as_time) nmv = nmv + 1  ! for un-doing all the extra contexts
	RETURN
 5050	buff1 = LEFINT( max_line_on_plot, slen  )
	CALL ERRMSG( ferr_invalid_command, status, 'cannot plot more than '
     .                  //buff1(:slen)//' lines with a single PLOT', *5000)
 5100	CALL ERRMSG( ferr_invalid_command, status, 
     .		     cmnd_buff(:len_cmnd)//' : vs what ?', *5000 )
 5110	CALL ERRMSG( ferr_invalid_command, status, 
     .		     ' overlay when no existing plot', *5000 )
 5120	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'overlay is on a different axis'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
 5130	buff1 = LEFINT( npts, slen  )
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
        
! 5150	buff3 = LEFINT( INT(0.999*pplmem_nsize), slen3 )
!	buff2 = LEFINT( plot_mem_used, slen2 )
!	CALL ERRMSG( ferr_prog_limit, status,
!     .          'Requested '//buff2(:slen2)//' words to plot'//pCR//
!     .		'Plot buffer size is: '//buff3(:slen3), *5000 )

 5160	cal_name = TM_GET_CALENDAR_NAME(cal_id_old)
        cal_name_new = TM_GET_CALENDAR_NAME(cal_id_new)
        slen = TM_LENSTR1 (cal_name)
        slen2 = TM_LENSTR1(cal_name_new)
	CALL ERRMSG( ferr_inconsist_grid, status,
     .          'Differing calendar axes: '//pCR//
     .		'first variable is on '//cal_name(:slen)//
     .          ' axis'//pCR//
     .		'subsequent variable is on '//cal_name_new(:slen2)//
     .		' axis', *5000 )

 5170	buff1 = LEFT_REAL (first, '(G15.3)', slen)
	buff2 = LEFT_REAL (last, '(G15.3)', slen2)
	CALL ERRMSG( ferr_out_of_range, status,
     . 'Limits for log axis negative or too small: '//
     .  buff1(:slen)// ' : '//  buff2(:slen2), *5000 )

 5171	CALL ERRMSG( ferr_out_of_range, status,
     . 'Data all-missing. Cannot define log axis for plotting',
     .  *5000 )


 5180	CALL ERRMSG( ferr_limits, status,
     . 'One-point independent axis: Requires a '//
     .  '/HLIMIT or /VLIMIT specification', *5000 )

 5190	CALL ERRMSG( ferr_syntax, status,
     . 'PLOT/VS/GAPLOC only for a single dependent variable', *5000 )

 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'Data grid is not a DSG grid ',
     .		*5000 )

 5210	CALL ERRMSG( ferr_grid_definition, status,
     .		'Plot trajectoryProfile data as trajectory, '//
     .          'only for a single variable',
     .		*5000 )
     
 5300   CALL ERRMSG( ferr_out_of_range, status,
     .     'data too large to plot'//pCR//
     .      cmnd_buff(:len_cmnd), *5100 )

	END
