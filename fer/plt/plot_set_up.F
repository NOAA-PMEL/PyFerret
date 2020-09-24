        SUBROUTINE PLOT_SET_UP( 
     .				overlay, transpz, versus, no_labels,
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
* (the routine is confusing as a result of simultaneously coping with the needs
* of single variables, multiple variables, /VS, /TRANSPOSE and /OVERLAY)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87  - based on DISP_SET_UP rev 1.1
* revision 0.1 - 7/21/87 - added transpose option and 
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
*       1/00 - *kob* - check the variable time_axis before looking at
*                      any time-relevant variables.
*       3/00 - *sh* added use_line and sym_size args for PLOT/COLOR=/SIZE=
*	     - pass no_labels to AXIS_FORMAT 
*	5/00 - added /HLIMITS and /VLIMITS (denigrating /xlimits/ylimits)
* V521	7/00 - *sh* bug fix to /TRANS/VLIMITS=... 
*	7/00 - *sh* new argument range_rqd needed on EQUAL_RANGE
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
* V530 *ACM* 1/01  cal_id for alternative calendar axes.
*      *sh* 1/01 - added argument step_inc for PLOT/STEP
*	         - bug fix: PLOT+ mem limit check off by x2 --  X&Y components
*      *acm* 5/01 - added  PLOT/DASH  (do_dash, dashstyle)
*      *acm* 5/01 - correct STEP bug: send NLOAD rather than NPTS2 to PPLLDX 
* V533 *acm* 6/01 - for LOG plots add logicals is_logx, is_logy.  set to plot
*                   log of depth as inverse axis
* V533 *acm* 9/01 - changes for dynamic pplus buffer.  Rename NSIZE to pplmem_nsize.
*                   call PPLLDX via pplldx_envelope.c which checks pplmem_nsize 
*                   and allocates more memory if needed.
*      *acm* 10/01  remove VMS includes
* V552 *acm*  3/03  Send the key string to key_loc since it may be multi-line
*                   also, make strings longer
* V580 *acm*11/04  Allow thickness to be set when cycling through colors automatically.
*                  In call to line_style, when color = -11  then thin line, color=-21 
*                  then thick=2, when color=-31 then thick=3
* V580 *acm*11/04  For /VS plots turn on lon/lat axis formatting, if lon/lat units
*                  (as in polygon_set_up).
* V580 *acm* 12/04 Fix bug 1129; formatting axis for small lon/lat ranges:
*                  send delta to routine AXIS_FORMAT
* V580 *acm* 10/05 Fix for bug 1349, longitude/latitude axis formatting. Need 
*                  to set the format to use as well as the label.
* V610 *acm*  2/08 Fix bug 1560. In the lon/lat axis formatting, check whether 
*                  the unit is longitude or latitude. PLOT/VS plots may have 
*                  either one of these on either axis.
* V612 *acm*  7/08 Prev. fix was that variables with units of "degrees_e" and 
*                  "degrees_n" get longitude/latitude units on a PLOT/VS plot. 
*                  Also allow units of "longitude" or "latitude" to trigger 
*                  lon/lat labels.
* V612 *acm*  8/08 Fixing bug 1038. No Valid Data labeled as on 2D plots. If 
*                  there is no valid data, send that information back, so a
*                  label can be added to the plot. If this is an overlay, add
*                  No Valid Data to the label.
* V614  *acm* 10/08 - add new MODE NODATA_LAB to control the NO VALID DATA label.
*                  Put it on plots only when this mode is set.
* V62   *acm*  2/09 Fix bug 1636, on PLOT/VS when setting up dependent axis,
*                  need to reinitialize delta=unspecified_val8 so that delta is
*                  computed from the dependent data.
* V662 *acm*  5/10 Set AX_VERT or AX_HORIZ symbol with the direction of the independent axis: X, Y, Z, T" plot_set_up.F
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret: Add E and F dimensions 
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP.
* V6.85 *acm* 9/13 New PLOT/COLOR w/o argument for all-color line plot
* V685+ *acm  12/13 In dependent axis setup, make sure variable the_taxis is set.
* V691  *acm*  5/14 For ribbon line plots, set up for histogram levels
* V691  *acm*  7/14 Fix ticket 2187: axis labels for PLOT/VS plots.
* V693  *acm* 10/14 Fix 2212: Check for a valid axis if they've asked for a 
*                   log axis before proceeding with vlimits or hlimits
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V695+ *sh*  3/15 if an FMRC dataset disambiguate T and F axes with labels
* V694  *acm*  5/15 Fix ticket 2282: setting up axis for ribbon plots
* v696  *acm*  6/15 PLOT/VS/GAPLOC to define seperate line segments on
*                   PLOT/VS plots (LAS ticket 1597 and others)
* V695  *acm* 9/15 send REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* V695  *acm* 9/15 Changes for HH:MM labeling when plotting short time range.
*                  For these axes reset the start time by only a day.
* V695  *acm* 9/15 Tweak to above - dont reset tref for overlay plots.
* V697  *acm* 1/16 Further fix ticket Fix 2122. Make sure the dependent axis
*                  data min and max are set.
* V698 *sh* 3/16 - limit max number of plot keys generated
* V698  4/16 *acm* New PLOT/SKIP=
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* V701  *acm* 8/16 Fix ticket 2463: single-valued data with a /LEVELS setting
* V701  *acm* 8/16 Fix ticket 2468: Symbols AX_HORIZ and AX_VERT are not reset on an /OVERLAY plot
* V702  *acm* 10/16 Test the results of MINMAX routine before use; if all data is 
*                   all-missing it returns arbitrary_large_val4.
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V710  *acm* 11/16 Ticket 2483: Fix the axis-end symbols that are defined 
*                   for time axis plots
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V702 *sh* 1/17 bug fix - ensure only 1D data goes to EXTRACT_LINE if not a
*                      PLOT/VERSUS
* V71  *acm* 2/17  More on ticket 2231. New argument to FULL_VAR_TITLE to control units labels
* V720 *acm* 3/2017 Ticket 2517: Bugs in PLOT/VS when the units are degrees. Axis formatting
*                   needs to be consistent and controlled by modes long_label and latit_label
* v7.2 *acm* 2/2017 ticket 2446 Drawing formatted time axes on a PLOT/VS when a variable has
*                   units of e.g. "days since datestring". The datestring can be in a recognized
*                   format: 01-jan-2011, or 2011-01-01. MM:HH:SS if present can be written
*                   separated by a space, or a colon, or Z.
* V7.2 *acm* 4/2017 Ticket 2318: for constant-value data on a line plot, make the axis
*                   range +/- 10% of the value instead of +/- 1. For very large or small
*                   values the +/- 1 corrupted the labels or the location of the plot.
* V7.2 *acm* 4/2017 In fix for 2212, make sure ind_min,ind_max are defined before using them 
*                   to check log limits on log axis.  Make sure the limits are actually applied.
* V7.2 *acm* 5/2017 restore correct longitude moduloing for PLOT/VS overlay plots
* V7.22 *acm* 10/2017 Fix ticket 2565, duplicate and sometimes shifted labels on PLOT/over time plots
* V73+ 1/18 *acm* Changes for Issue 1009; precsision in time info sent to pplus for 2D plots
*                 handle line plots over scaled 2D plot
*                 On time plots overlaid on a 2D plot, set up time-axis information even when
*                 not labeling the plot.
* V7434 10/18 *acm* minor cleanup: initialize variable the_taxis
* V74+ *acm*  4/18 additions for plotting trajectories from a dsg dataset
* V74+ 4/18 *acm*  Automatically do a colored trajectory plot for the command 
*                  "PLOT var" if it's a DSG trajectory set. 
* V74+ 4/18 *acm*  Updates to DSG plotting: trajectories or sets of timeseries etc
* V74+ 7/18 *acm*  Updates to DSG plotting: 
*                   - Fixes for modulo longitude ranges in the context
*                   - Upper-left labels with plot range in world coordinates
* V7.4+ *acm*  8/18 DSG-enabled Ferret: line-key labels may be strings from the IDs
* V7.4+ *acm* 10/18 Handle single-point plots: plot a symbol
* V7.5+ *acm*  4/18 PLOT/VS plots from DSG data that are not native-trajectory plots
* V75   10/18 *acm* ticket 1926 F-axis line plots were trying to get time-origin from t-axis
*                   of the grid.
* V75    4/19 *acm* For data in profile DSG dataset, 2- or 3-argument ribbon plot
*                   may need an argument expanded to the obs length, e.g ribbon/vs time, depth, var
* V7.5+ *acm*  6/19 changes to the call of PLOT_DSG_CHECK_LON360
* V751   6/19 *acm* Issue 1940, dependent variable on a time axis with /vlimits.  The call to
*                   EQUAL_RANGE specified that the dependent-axis limits be translated as dates 
*                   on the independent time axis.
* V751   6/19 *acm* For DSG data update calls to PLOT_DSG_APPLY_OBS_MASK
* V751   7/19 *acm* For DSG data, set a flag controling a PLOT/VS plot with station locations as
*                   opposed to exanding variables to the obs axis.
* V751 7/19 *acm*  DSG-enabled Ferret: plotting user-defined variables
* V7.6  *acm*  1/20 Set up for plotting Points-type datasets as ribbon plots
*                   with symbols. 
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
* V760 *acm* 3/20 Customize plot for case when constraints result in no features
* V760 *acm* 4/20 PLOT_DSG_CHECK_LON360 now applies a lon-360 conversion of the 
*                 longitudes if it'll be better and /HLIM not given
* V760 *acm* 4/20 Issue 1973: For PLOT/VS when the units are degrees lon or lat, 
*		  mark the axis direction info in plot_axes( ) so that overlays 
*                 such as"go fland" will work. (DSG and non-DSG data.)
* V760 *acm* 4/20 Similarly, if PLOT/VS and a depth coord var is on the vertical
*                 axis, reverse the axis to run DEEP:SHALLOW
* V762 *acm* 8/20 Add a second test for is this a trajectory ribbon plot.
* v763 *acm* 9/20 Plots of id-variables in trajectory-profile, timeseries-profile data

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
	include 'xdyn_linemem.cmn_text'


#include "tmap_dset.parm"

* calling argument declarations:
	LOGICAL		overlay, transpz, versus, no_labels, use_line,
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
     .			flip, time_axis,
     .			all_1_ind, formatted,
     .			indep_is_log, dep_is_log, use_nice,
     .                  this_no_range(max_line_on_plot), set_axis,
     .                  timeax_both, do_units, doing_ribbon_color,
     .                  x_is_time, y_is_time, its_traj, its_dsg, its_cmpnd,
     .                  expand_dsg_ok, color_dsg_id, user_hlim, dsg_as_time, 
     .                  dsg_as_traj, do_traj

	CHARACTER*2040	FULL_VAR_TITLE, KEY_STRING, plot_title
	CHARACTER	INTERNAL_WHOI_DATE*14, VAR_UNITS*64,
     .			SECS_TO_DATE*20, LEFINT*8, MERGED_WHOI_DATE*14, 
     .			LEFT_REAL*16, TM_GET_CALENDAR_NAME*32, 
     .			indep_ax*1, dep_ax*1, tstyle*3, xtra_lab*24,
     .			buff1*40, buff2*16, buff3*8, t1_date*14, tref*14,
     .                  cal_name*32, cal_name_new*32, val_buff*45, fmt*4,
     .			whoimin*14, whoimax*14, 
     .			whoimin_axlo*14, whoimax_axhi*14
	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_SIZE, CGRID_AXIS, 
     .			TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
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
     .			dsg_fmt_grid, dep_dir, nmasked, dset_dsg, dum1, dum2

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

	  
* limits error check
        IF (nmv .GT. max_line_on_plot) GOTO 5050

	IF (versus) THEN
	   CALL PLOT_VS_SET_UP ( 
     .				overlay, transpz, no_labels,
     .				symbol, sym_size, color, color1, use_line,
     .				step_inc, do_dash, dashstyle,
     .                          is_logx, is_logy, nokey, addgaps,
     .				all_1_dep, only_val, skipsym, mv_list, 
     .				cx_list, nmv, indep_dat, dep_dat, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	   GOTO 1000
	ENDIF

	
* initialize

	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	cx   = cx_list( 1 )
	mv   = mv_list( 1 )
	grid1 = cx_grid( cx )
	grid = cx_grid(is_cx(1))
	dset = cx_data_set(cx)
	indep_dim = plot_axis(1)
	dsg_as_time = .FALSE.

	numv = nmv  ! local copy
	line_symbol = symbol
	IF (cxdsg_empty_set) THEN
	   sym_size = 0.001
	   line_symbol = 1        ! doesn't matter what
	   use_line = .FALSE.
	ENDIF


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
	    grid = cx_grid(is_cx(nmv))
	    IF (dset_dsg .LT. pdset_irrelevant) grid = dsg_xlate_grid(dset)

	    nfeatures = TM_DSG_NFEATURES( grid )	    

* The dataset may be DSG but the variable not.
	    CALL TM_DSG_FACTS( grid, orientation, obsdimlen, fline, its_dsg, its_cmpnd )
 
	    CALL DSG_PLTALONG_SETUP (dset, orientation, do_traj, 
     .			dsg_as_traj, dsg_as_time, status)
	    IF (status .NE. ferr_ok) GOTO 5000

* Is this just a regular ribon plot with 3 variables (not a lon/lat traj plot)? 
* OR a PLOT/VS var1,var2 from a DSG dataset for var1,var2 not lon/lat?
* This may turn off ribbon_plot

	    IF (.NOT.dsg_as_traj .AND. .NOT.dsg_as_time) 
     .	         CALL CHECK_PLOT_TRAJ (versus, numv, grid, its_traj)
	    its_traj = its_traj .OR. do_traj

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

	       IF (do_traj .OR. dsg_as_traj) THEN
		  CALL  DSG_TRAJ_ALONGXY_SETUP( dset, 
     .                 overlay, no_labels, symbol, sym_size, color, 
     .                 color1, use_line, do_dash, dashstyle, nokey,
     .                 only_val, skipsym, mv_list, cx_list, 
     .                 indep_dat, dep_dat, status )

	       ELSE

		  its_traj = dsg_feature_type(dset) .EQ. pfeatureType_Trajectory
		  CALL DSG_TRAJ_PLOT_SET_UP ( dset,
     .                overlay, no_labels, symbol, sym_size, color, color1, 
     .                use_line, do_dash, dashstyle, nokey, its_traj, only_val, 
     .                skipsym,  mv_list(1), cx_list(1), numv, indep_dat, dep_dat, status )

		  IF (numv.EQ.3 .AND. status.NE.ferr_ok) GOTO 2000  ! not traj after all, just a ribbon plot
	       ENDIF

	       GOTO 1000

	    ENDIF  ! its_traj

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
        DO 10 ipl = 1, numv
 10     plot_mem_used = plot_mem_used
     .		      + 2*step_inc*CGRID_SIZE( cx_list(ipl) )

* check for improper data supplied
	IF ( overlay ) THEN
	   CALL GET_CX_DIMS( cx, ndim, dims )
           IF ( twodee_on ) indep_dim = dims(1)       ! 3/93
	   IF ( ndim.NE.1 .OR. dims(1).NE.indep_dim ) GOTO 5120
	ELSE
	   ndim = nplot_axis
	   IF ( ndim .GT. 1 ) GOTO 5110
	ENDIF

* length of plot array
	
	npts = CX_DIM_LEN ( indep_dim, cx )

* Overlay plots, keep the setting of IAUTOT that was set on the underlay plot.

	IF (overlay) THEN
	   iautot_save = iautot
	   iautot = 0
	ENDIF

* set flag indicating a 1D plot is on the screen
	IF (.NOT.overlay) onedee_on = .TRUE.

	IF (ribbon_plot) ribbon_var = 2

* decide if the picture should be rotated 90 degrees
	IF ( indep_dim .EQ. z_dim  ) THEN
* ... flip depth plots by default
	   flip = .NOT.transpz
	ELSE
	   flip = transpz
	ENDIF
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

* define special axis labels to disambiguate T and F calendar axes
           IF ( ITS_FMRC(grid1) ) THEN
	      IF (GEOG_LABEL(indep_dim,grid1)) THEN
	         IF     (indep_dim .EQ. t_dim) THEN
	            xtra_lab = 'Verification Time ==>'
	         ELSEIF (indep_dim .EQ. f_dim) THEN
	            xtra_lab = 'Initialization Time ==>'
	         ENDIF
	      ENDIF
	   ENDIF

	   var1 = 1
	   ndv  = numv
	   ito  = 0
	   time_axis =	indep_dim.EQ.t_dim .OR. indep_dim.EQ.f_dim
	   IF ( time_axis ) THEN	! 11/95: are ALL vars time vars?

	      the_taxis = indep_dim

* and are all the calendar types the same?

              tax = grid_line(the_taxis,cx_grid(cx_list(var1)) )
              cal_name = line_cal_name(tax)
              cal_id_1 = TM_GET_CALENDAR_ID ( cal_name )

	      DO 100 ipl = var1, numv
	         time_axis = time_axis
     .			.AND. GEOG_LABEL( the_taxis, cx_grid(cx_list(ipl)) )
                 tax = grid_line(the_taxis,cx_grid(cx_list(ipl)) )
                 cal_id_2 = TM_GET_CALENDAR_ID ( line_cal_name(tax) )
                 IF (cal_id_2 .NE. cal_id_1) THEN
                    cal_id_old = cal_id_1
                    cal_id_new = cal_id_2
                    GO TO 5160
                 ENDIF
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

	      ppl_buff = ' '
	      WRITE ( ppl_buff, 3004 ) cal_name
              CALL PPLCMD (from, line, 0, ppl_buff, 1, 1)

c	      ul_dolab(t_dim) = .FALSE.  ! why was this added?
	   ENDIF


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

	DO 200 ipl = var1, numv
	   
	   mv   = mv_list( ipl )
	   cx   = cx_list( ipl )
	   grid = cx_grid( cx )
	   dset = cx_data_set(cx)
           CALL GET_CX_DIMS( cx, ndim, dims )
	   IF (ndim .GT. 1) GOTO 5110

* set up a dependent variable.

	   IF (dsg_as_time) THEN
	      npts2 = CX_DIM_LEN (indep_dim, cx)
	      dum1 = 0
	      DO i = cx_lo_ss(cx,indep_dim), cx_hi_ss(cx,indep_dim)
	         dum1 = dum1+1
	         dep_dat(dum1) = dsg_linemem(cxdsg_lm_tsdat)%ptr(i)  
	      ENDDO
	      dep_dim = t_dim
	      
	   ELSE

	      CALL EXTRACT_LINE (	cx,
     .				memry(mv)%ptr,
     .				mv,
     .				dep_dat,
     .				dep_dim,
     .				ndim,
     .				npts2,
     .				status	)

	      iaxis = CGRID_AXIS(dep_dim,cx)
	   
	   ENDIF

	   IF ( status .NE. ferr_ok ) GOTO 5000

	   IF (cxdsg_empty_set) THEN
	      dep_dat(1) = -1.
	      dep_dat(2) =  1.
	   ENDIF
	      

* ... If this is a profile or timeseries collection maybe we are just makinga map with 
*     the stations so dep_dat and indep dat are the longitudes and latitudes. The flag
*     expand_dsg_ok false, says just plot the instance variables

* ... For a dsg dataset, can ask to color by a feature-level variable
*     such as an ID or a string-valued feature variable. 

	   doing_ribbon_color = ribbon_plot .AND. ipl.EQ.ribbon_var

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

* ... in PLOT TEMP , SALT , U all vars must be along same ww dimension

	   IF ( ndim .GT. 1 ) GOTO 5110
	   IF ( dep_dim .NE. indep_dim ) GOTO 5140
	   CALL MAKE_STEP_AXIS( grid, dep_dim, step_inc,
     .			cx_lo_ss(cx,dep_dim), cx_hi_ss(cx,dep_dim),
     .			indep_dat )
	   IF ( time_axis ) THEN
	      ind_min = MIN( ind_min,
     .                     TSTEP_TO_SECS(grid, the_taxis, DBLE(indep_dat(1)) ))
	      ind_max = MAX( ind_max,
     .                     TSTEP_TO_SECS(grid, the_taxis, DBLE(indep_dat(nload)) ))

* Plotting short time range? If so recompute tref     
* axis length in hours from time since BC in seconds
	      IF (.NOT.overlay) THEN
	         hrs = ( ind_max - ind_min ) / 3600.
	         IF (hrs.LT.24) tref = 
     . 	            MERGED_WHOI_DATE( cx_list(var1), the_taxis, numv, .TRUE. )
	      ENDIF
	   ELSE
	         
	      CALL MINMAX( indep_dat, npts, mr_bad_data(mv), lo, hi, nok )
	      ind_min = MIN(ind_min, lo)
	      ind_max = MAX(ind_max, hi)

	   ENDIF

	   all_1_ind = ind_min .EQ. ind_max
	   IF ( all_1_ind ) THEN
	      delta = 1.0
	      IF (ind_min .NE. 0) delta = 0.1* ind_min
	      IF (indep_dim .EQ. the_taxis) delta = 240*3600
	      ind_min = ind_min - delta
	      ind_max = ind_max + delta
	   ENDIF

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
              ENDIF
              t1_date = INTERNAL_WHOI_DATE( grid, the_taxis, 1.0D0 )
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

* See pplldx comments. When loading the color-by variable lon vertical time axis
* this forces the right combination of settings.

	   IF (doing_ribbon_color) icode = 2
	   IF (y_is_time .AND. doing_ribbon_color) icode = 7

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
	      
	      delta = unspecified_val8

	      IF ( pindeplim .GT. 0 ) THEN
	         IF (pxlim.GT.0 .AND. .NOT.denig_xylim_msg_done) THEN
	            CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	            CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	            denig_xylim_msg_done = .TRUE.
	         ENDIF
	         CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pindeplim):qual_end(pindeplim)),
     .			indep_dim, ind_min, ind_max, delta,
     .			formatted, range_rqd, cal_id_1, status )
	         IF ( status .NE. ferr_OK ) GOTO 5000
		 
*  Check for valid log axis before proceeding
	         IF (indep_is_log) THEN
	            CALL AXIS_ENDS  ( indep_ax, indep_dim, grid1,
     .				   ind_min, ind_max, delta, 
     .				   indep_is_log, indep_axtyp, versus, status )
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

* convert time given in time steps to absolute units	   
	         IF ( time_axis .AND. .NOT.formatted ) THEN
* time limits given as tstep range although mode calendar is true
	            IF ( ind_min .GT. ind_max ) THEN  ! no reversed calendar
	               tmp = ind_min
	               ind_min = ind_max
	               ind_max = tmp
	            ENDIF
	            ind_min = TSTEP_TO_SECS( grid1, the_taxis, ind_min )
	            ind_max = TSTEP_TO_SECS( grid1, the_taxis, ind_max )
	         ELSEIF ( indep_dim .EQ. the_taxis
     .		   .AND. .NOT.time_axis
     .	           .AND.  formatted ) THEN
* time limits given as dates although mode calendar is false
	            ind_min = SECS_TO_TSTEP( grid1, the_taxis, ind_min )
	            ind_max = SECS_TO_TSTEP( grid1, the_taxis, ind_max )
	         ENDIF
              ELSE   ! /HLIM or /VLIM not used; check for range in indep direction

c one-point on the independent axis, still make a nominal plot?
c                 IF (nload_all .EQ. 1) GOTO 5180

                 IF (nload_all .EQ. 1) THEN
		    IF ( indep_dim .EQ. the_taxis) THEN  ! nominally +/- 10 day
c		       ind_min = indep_dat(1) - 240*3600
c		       ind_max = indep_dat(1) + 240*3600
		       pindeplim = ind_max - ind_min
		    ELSE
		       ind_min = ind_min - 1.
		       ind_max = ind_max + 1.
		       pindeplim = 2.
		    ENDIF
		    IF (line_symbol.EQ.qual_off .OR. 
     .               line_symbol.EQ.unspecified_int4) 
     .               line_symbol = qual_on  ! draw point with a symbol
		    CALL LINE_STYLE(line_symbol, sym_size, skipsym, color, color1, use_line,
     .			   do_dash, dashstyle, nline_in_mem, 1)
		    
                 ENDIF

	      ENDIF
	      IF ( time_axis ) THEN
	         IF ( pindeplim .GT. 0 ) CALL PPL_TIME( ind_min,ind_max,
     .                                    cal_id_1)
	         CALL TAXIS_STYLE( indep_ax, ind_min, ind_max, tstyle, xtra_lab )
* (6/99) Note: The time axis extrema symbols are only approximately correct
*		since PLOT+ rounds out the time ranges ...

* 11/16 Get the min and max as used by PPLUS.  That function uses WHOI formatted time
	         CALL TPLOT_AXIS_ENDS (ind_min, ind_max, cal_id_1, tstyle)

	         CALL AXIS_END_SYMS( indep_ax,
     .				SECS_TO_TSTEP( grid1, the_taxis, ind_min ),
     .				SECS_TO_TSTEP( grid1, the_taxis, ind_max ) )
	      ELSE
	         CALL AXIS_ENDS  ( indep_ax, indep_dim, grid1,
     .				   ind_min, ind_max, delta, 
     .				   indep_is_log, indep_axtyp, versus, status )
	         IF ( status .NE. ferr_OK ) THEN
                    first = ind_min
                    last  = ind_max
                    GOTO 5170
                 ENDIF
	
	         timeax_both = .FALSE.
	         CALL AXIS_FORMAT( indep_ax, indep_dim, grid1,
     .				   ind_min, ind_max, delta,
     .				   pnot_curv, 0,
     .				   time_axis, timeax_both, tstyle,
     .                             no_labels, xtra_lab, ascale, aoff )
	      ENDIF

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
	      IF (val1 .NE. 0) delta = 0.1* val1
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid,val1-delta,val1+delta,delta,
     .                        dep_is_log, dep_axtyp, versus, status )   

	   ELSEIF (dep_is_log) THEN	! get dependent var scaling and set up log axis
	      CALL MINMAX( dep_dat, npts, mr_bad_data(mv), lo, hi, nok )
	      IF (nok .EQ. 0) GOTO 5171  ! all missing
	      dep_min = lo
	      dep_max = hi
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, dep_is_log, dep_axtyp, versus, status )
     
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

	   fvar = 0
	   IF (dset .GT. pdset_irrelevant) fvar = dsg_feature_var(dset)
	   IF (dsg_as_time) fvar = dsg_ftrset_var(dset)
	   
	   CALL LINE_PLOT_LABELS (var1, nkey_entries, ndv, cx_list, 
     .		this_no_range, overlay, versus, nokey, time_axis, 
     .		tstyle, cal_id_1, ribbon_var, indep_lab, dep_lab, 
     .		ind_min, dep_len, hlen, nfeatures, dset, fvar)
	ENDIF

* successful completion
 1000	CONTINUE
	status = ferr_ok
	IF (overlay) iautot = iautot_save
	IF (dsg_as_time .AND. cxdsg_lm_tsdat.NE.mnormal) THEN
	   CALL TM_FREE_LM_DYNMEM (cxdsg_lm_tsdat)
	   cxdsg_lm_tsdat = mnormal
	ENDIF
	
	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
 	IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
	IF (dsg_as_time .AND. cxdsg_lm_tsdat.NE.mnormal) THEN
	   CALL TM_FREE_LM_DYNMEM (cxdsg_lm_tsdat)
	   cxdsg_lm_tsdat = mnormal
	ENDIF
	RETURN
 5050	buff1 = LEFINT( max_line_on_plot, slen  )
	CALL ERRMSG( ferr_invalid_command, status, 'cannot plot more than '
     .                  //buff1(:slen)//' lines with a single PLOT', *5000)
 5100	CALL ERRMSG( ferr_invalid_command, status, 
     .		     cmnd_buff(:len_cmnd)//' : vs what ?', *5000 )
 5110	dep_ax = LEFINT( ndim, slen )
	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'specified data is not a line'//pCR//
     .			' - its a '//dep_ax(:slen)//'D region: "'//
     .			cmnd_buff(:len_cmnd)//'"', *5000 )
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
	END
