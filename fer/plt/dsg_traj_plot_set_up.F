        SUBROUTINE DSG_TRAJ_PLOT_SET_UP( 
     .				overlay, no_labels, symbol, sym_size, color, 
     .				color1, use_line, do_dash, dashstyle, nokey,
     .				is_traj, only_val, skipsym, mv_list, 
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

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG
*
* based on plot_set_up, Set up to draw trajectory plots from a dsg trajectory dataset
* v7.4+ *acm* 7/2018 color-by trajectory, even if the trajectory variable is a string.
* v7.4+ *acm* 7/2018 set up colorbar as for ribbon plots
* V7.4+ *acm*  8/18 DSG-enabled Ferret: line-key labels may be strings or numeric values
*                    from a feature-direction metadata variable.
* V7.4+ *acm* 10/18 Handle single-point trajectories: plot a symbol
* V7.5+ *acm*  6/19 Better handling of plotting longitudes without branch jumps.  
*                   Do masking correctly.
* V7.51 *acm*  8/19 getting min/max of longitudes and latitudes
* V7.51 *acm*  8/19 Ticket 1953, masking and min/max of dependent axis data
* V7.6  *acm*  1/20 Set up for plotting Points-type datasets as ribbon plots
*                   with symbols. Flag is_traj indicates trajectory data type
* V7.6  *acm*  3/20 Ignore PLOT/LEVELS=v when color=by the id_variable. This 
*                   doesn't make sense and makes a screwy plot.
* V760 *acm* 3/20 Customize plot for case when constraints result in no features
* V760 *acm* 4/20 PLOT_DSG_CHECK_LON360 now applies a lon-360 conversion of the 
*                 longitudes if it'll be better and /HLIM not given
* V760 *acm* 5/20 initialize dep_dim. Was unset when we get latitude from the DSG dataset

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
	include 'xdsg_context.cmn'
	include 'xvariables.cmn'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
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

* calling argument declarations:
	LOGICAL		overlay, no_labels, use_line,
     .			nokey, is_traj
	INTEGER		symbol, color, color1, do_dash, skipsym,
     .			nmv, mv_list( nmv ), cx_list( nmv ), status
	REAL		sym_size, dashstyle(*), only_val
	REAL		indep_dat(*), dep_dat(*)     ! dynamic space allocation
 
* V500 *kob* 3/99- up VAR_CODE and VAR_UNITS to 64 chars
* internal variable declarations:

	LOGICAL		NO_LINE_RANGE, GEOG_LABEL, TM_ITSA_DSG_RAGGED,
     .			versus, all_1_ind, all_1_dep, formatted, use_nice, 
     .			this_no_range(max_line_on_plot), indep_is_log, set_lev, 
     .			its_dsg, its_cmpnd, mask_applied, user_hlim
	CHARACTER	LEFINT*8, LEFT_REAL*16, FULL_VAR_TITLE*2040,
     .			indep_ax*1, dep_ax*1, tstyle*3, 
     .			buff1*40, buff2*16, t1_date*14, tref*14, fmt*4
	INTEGER		TM_LENSTR1, CGRID_SIZE, CGRID_AXIS, 
     .			TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, TM_DSG_OBS_GRID,
     .			mv, cx, slen, ito, indep_lab, dep_lab, nline_in_mem,
     .			indep_dim, npts, dep_dim, npts2, ndv, var1,
     .			grid, grid1, ipl, ndim, dims(6),
     .			slen2, slen3, tax, plot_mem_used, i,
     .			pxlim, pylim, phlim, pvlim, pindeplim, pdeplim,
     .			dep_axtyp, indep_axtyp, the_taxis, ribbon_var, 
     .			gap_var, iiunits, iaxis, hblk1, mvh_temp, 
     .			nkey_entries, itmp, igrd_save, cal_id_1,
     .			icode, dset, nfeatures, obsdimlen, fline,
     .			idim, coord_lm, ivar, ptitl, dim(nferdims), numcx,
     .			dep_nok, indep_nok, line_symbol, nmasked 

        INTEGER         iunits, junits
	CHARACTER*1     axdir(nferdims), ax1

	REAL		val1, dt_min, lo, hi, dep_len, 
     .			ind_len, first, last, hlen, dep_lo, 
     .			dep_hi, indep_lo, indep_hi
	REAL		ind_min, ind_max, dep_min, dep_max, 
     .			delta, tmp, labsize
	REAL*4 rbad

* local parameter declarations:
	LOGICAL		range_rqd
	PARAMETER     ( range_rqd = .TRUE. )
	DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'setting up plot', 15)

* initialize

	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	versus = .TRUE.
	cx   = cx_list( nmv )
	mv   = mv_list( nmv )
	indep_dim = plot_axis(1)
	dep_dim = plot_axis(2)
	nline_in_mem = 0
	phlim = qual_given( slash_hlimits )
	pvlim = qual_given( slash_vlimits )
	pxlim = qual_given( slash_xlimits )	! deprecated
	pylim = qual_given( slash_ylimits )	! deprecated
	dep_axtyp = 1
	indep_axtyp = 1
	indep_is_log = .FALSE.
	mod_vs_x = .FALSE.
	mod_vs_y = .FALSE.
        nkey_entries = nmv
	itmp = mnormal
	icode = 0
	ribbon_var = 1
	hlen = xlen 
	line_symbol = symbol

	! dummy values
	cal_id_1 = 0
	t1_date = ' '
	tref = ' '

* For DSG dataset get number of features and obsdimlen

	grid1 = cx_grid( cx )

	IF (  qual_given(slash_plot_along) .GT. 0 )  
     .	       CALL WARN ('PLOT/ALONG is ignored for discrete DSG data')
	grid = cx_grid(is_cx(nmv))
	dset = TM_DSG_DSET_FROM_GRID( grid )
	nfeatures = TM_DSG_NFEATURES( grid )
	its_dsg = .TRUE.    
	CALL TM_DSG_FACTS( grid, idim, obsdimlen, fline, its_dsg, its_cmpnd )
	IF (.NOT. its_dsg) GOTO 5200

* The variable may be on the feature-axis. We will deal with that later but
* for getting the lon/lat data we need the obs grid.

         IF ( .NOT. TM_ITSA_DSG_RAGGED( grid ) ) THEN
	    indep_dim = idim
	    grid1 = TM_DSG_OBS_GRID(dset)
	 ENDIF
 

* check for improper data supplied
	IF ( overlay ) THEN
	   CALL GET_CX_DIMS( cx, ndim, dims )
           IF ( twodee_on ) indep_dim = dims(1)       ! 3/93
	   IF ( ndim.NE.1 .OR. dims(1).NE.indep_dim ) GOTO 5120
	ELSE
	   ndim = nplot_axis
	   IF ( ndim .GT. 1 ) GOTO 5110
	ENDIF

* Overlay plots, keep the setting of IAUTOT that was set on the underlay plot.

	IF (overlay) THEN
	   iautot_save = iautot
	   iautot = 0
	ENDIF

	dep_ax    = 'Y'
	indep_ax  = 'X'
	dep_lab   = ppl_ylab
	indep_lab = ppl_xlab
	ind_len   = xlen
	dep_len   = ylen
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

* do not re-draw the main label, logo, other labels from the underlay plot
	IF (overlay) THEN
	   nlabs_on = 0  
	   CALL PPLCMD ( from, line, 0, 'LABS ', 1, 1 )	
	ENDIF

	IF ( no_labels ) CALL PPLCMD ( from, line, 0, 'LABS ', 1, 1 )

	var1 = 1        ! The variable sent in is the color-by variable.
	ndv  = 1	! # of dependent variables

* * * * * * *    - - - INDEPENDENT AXIS DATA - - - * * * * * * * * *
	
* Get the LONGITUDE coordinate data

	idim = 1
	iaxis = grid_line(idim, dsg_xlate_grid(dset))
	iunits = line_unit_code ( iaxis )
	IF (iunits .NE. 4) GOTO 5130  ! must be longitude coordinates

	IF ( line_dim (iaxis) .NE. obsdimlen) GOTO 5140 

	ivar = dsg_coord_var(idim,dset)
	coord_lm = dsg_loaded_lm(ivar)  ! line memory table indices

	DO i = 1, obsdimlen
	   indep_dat(i) = dsg_linemem(coord_lm)%ptr(i)
	ENDDO

	npts = obsdimlen

	IF (nmv .GT. 1) THEN
	   cx = cx_list(1)
	   mv = mv_list(1)
	   indep_dim = plot_axis(1)
	   CALL EXTRACT_LINE ( cx,
     .                      memry(mv)%ptr,
     .                      mv,
     .                      indep_dat,
     .                      indep_dim,
     .                      ndim,
     .                      npts,
     .                      status)
	   IF (npts .NE. obsdimlen) THEN
	      status = 0
	      GOTO 5140
	   ENDIF
	   numcx = 1  ! The context should come from the color-by var
	   CALL GRP_CX_DOC( cx_list(nmv), numcx, cx_plot, key_doc )
	ENDIF	

*  Check for longitudes crossing the dateline. Looks only at location data in features
*  included in any feature-level mask, which is applied in PLOT_DSG_APPLY_OBS_MASK

	user_hlim = pindeplim .GT. 0
	CALL PLOT_DSG_CHECK_LON360 (dset, cx_list(nmv), indep_dat, bad_val4, 
     .                    npts, nfeatures, ind_min, ind_max, user_hlim)

* ... Apply an obs mask if any - this will cause the right range to be set on the
*     lon/ lat axes

	cxdsg_empty_set = .FALSE.

	CALL PLOT_DSG_APPLY_OBS_MASK (dset, cx_list(nmv), nfeatures, obsdimlen, 
     .		   indep_dat, nmasked)
     

! Or set npts = 2, and use cx_lo_ww and cx_hi_ww if they are the requested constraints...

	IF (nmasked .EQ. 0) THEN
	   cxdsg_empty_set = .TRUE.
	   cxdsg_orient = dsg_orientation(dset)

	   indep_nok = 0
	   nmasked = 2
	   sym_size = 0.001
	   nokey = .TRUE.
	   qual_given(slash_plot_nokey) = 1
	   nokey = .TRUE.

	   delta = unspecified_val8
	   ind_min = cx_lo_ww(x_dim, cx)
	   ind_max = cx_hi_ww(x_dim, cx)

	   ! Check against dsg lon coord range usin MODSCAT as in MAKE_DSG_OBS_MASK

	   indep_dat(1) = bad_val4 
	   indep_dat(2) = bad_val4
	ELSE
	   mask_applied = nmasked .LT. npts
	   CALL MINMAX( indep_dat, nmasked, bad_val4, ind_min, ind_max, indep_nok )

* ... force axis scaling if the data has no range
	   all_1_ind = NO_LINE_RANGE( indep_dat, nmasked, mr_bad_data(mv), val1 )
	   delta = unspecified_val8

	   IF ( pindeplim .GT. 0 ) THEN
	      IF ( pxlim.GT.0 .AND. .NOT.denig_xylim_msg_done ) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF

              CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pindeplim):qual_end(pindeplim)),
     .			indep_dim, ind_min, ind_max, delta,
     .			formatted, range_rqd, cal_id_1, status )

              IF ( status .NE. ferr_OK ) GOTO 5000

	   ELSEIF ( all_1_ind .AND. .NOT.overlay ) THEN
	      IF ( val1 .EQ. mr_bad_data(mv) ) THEN
	         delta = 1.
	         ind_min = 0.D0
	         ind_max = 1.D0
	      ELSE
	         delta = 1.
	         ind_min = val1 - delta
	         ind_max = val1 + delta
	      ENDIF
	   ELSE
	      CALL MINMAX( indep_dat, nmasked, bad_val4, lo, hi, indep_nok )
	      IF (indep_nok .EQ. 0) THEN ! all missing
	         ind_min = -1.
	         ind_max =  1.
	      ELSE
	         ind_min = lo
	         ind_max = hi
	      ENDIF
	   ENDIF

	ENDIF   ! cxdsg_empty_set 

* Set up independent axis for plotting

	IF (.NOT.overlay) THEN

	   CALL AXIS_ENDS(indep_ax,indep_dim,grid1,ind_min,ind_max,
     .                       delta, indep_is_log, indep_axtyp, versus, status)

	   IF ( status .NE. ferr_ok ) THEN
              first = ind_min
              last  = ind_max
	      GOTO 5170
	   ENDIF

* This routine checks the units and the setting for formatted lon/lat axes
* If the formatting has been turned off, resets iunits and flag mod_vs_x.

	   CALL GET_AXIS_FORMAT( ind_min, ind_max, delta, fmt, use_nice )
	   IF (cxdsg_empty_set) use_nice = .FALSE.
	   IF (use_nice) THEN
	      ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                   ',''''LONE'''')'
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	      CALL LON_LAT_FMT (x_dim, 'X')
	      mod_vs_x = .TRUE.
	   ENDIF  
		  
           CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 ) ! for now, anyway	
	      
        ELSE ! overlay
           CALL PPLCMD ( from, line, 0, indep_ax//'FOR 0', 1, 1 )

           mod_vs_x = .TRUE.
           mod_vs_y = .FALSE.

	ENDIF

* set up PLOT5 to ignore bad data flag
	ppl_buff = ' '
	WRITE ( ppl_buff, 3005 ) bad_val4, indep_ax
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1)

 3005	FORMAT ( 'LIMITS ',G15.8,1X,A1,'EQ' )

	ind_min = val1 ! used in the case of only 1 valid data

* * * * * * *    - - - LATITUDE DATA - - - * * * * * * * * *

	IF (cxdsg_empty_set) THEN
	   npts2 = 2
	   dep_nok = 0
	   dep_min = cx_lo_ww(y_dim, cx)
	   dep_max = cx_hi_ww(y_dim, cx)
	   dep_dat(1) = bad_val4
	   dep_dat(2) = bad_val4

	ELSE

	   idim = 2
	   iaxis = grid_line(idim, dsg_xlate_grid(dset))
	   iunits = line_unit_code ( iaxis )
	   IF (iunits .NE. 4) GOTO 5130  ! must be latitude coordinates

	   IF ( line_dim (iaxis) .NE. obsdimlen) GOTO 5140 

	   ivar = dsg_coord_var(idim,dset)
	   coord_lm = dsg_loaded_lm(ivar)  ! line memory table indices

	   DO i = 1, obsdimlen
	      dep_dat(i) = dsg_linemem(coord_lm)%ptr(i)
	   ENDDO

	   all_1_dep = .TRUE.
	   only_val  = bad_val4

	   IF (nmv .GT. 1) THEN
            mv   = mv_list( 2 )
            cx   = cx_list( 2 )
            CALL EXTRACT_LINE (  cx,
     .                          memry(mv)%ptr,
     .                          mv,
     .                          dep_dat,
     .                          dep_dim,
     .                          ndim,
     .                          npts2,
     .                          status  )
	      IF (npts .NE. obsdimlen) THEN
	         status = 0
	         GOTO 5140
	      ENDIF
	   ENDIF

* ... Apply an obs mask if any - this will cause the right range to be set on the
*     lon/ lat axes

	   IF (mask_applied) CALL PLOT_DSG_APPLY_OBS_MASK (dset, cx_list(nmv), 
     .		   nfeatures, obsdimlen, dep_dat, nmasked)

* Set up the dependent axis.

	   all_1_dep = NO_LINE_RANGE( dep_dat, nmasked, bad_val4, val1 )
	   CALL MINMAX( dep_dat, nmasked, bad_val4, dep_lo, dep_hi, dep_nok )
	   IF (dep_nok .EQ. 0) THEN ! all missing
	      dep_min = -1
	      dep_max =  1
	   ELSE
	      dep_min = dep_lo
	      dep_max = dep_hi
	      IF (all_1_dep) THEN
	         delta = 1.
	         dep_min = dep_lo - delta
	         dep_max = dep_hi + delta
	      ENDIF
	   ENDIF

	ENDIF ! cxdsg_empty_set


	IF (.NOT.overlay) THEN

	   delta = unspecified_val8    
	   CALL AXIS_ENDS(dep_ax, dep_dim, grid1, dep_min, dep_max,
     .                      delta, indep_is_log, dep_axtyp, versus, status)

	   CALL GET_AXIS_FORMAT( dep_min, dep_max, delta, 
     .                 fmt, use_nice )
	   IF (cxdsg_empty_set) use_nice = .FALSE.
	   IF (use_nice) THEN
	      ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .                       ',''''LAT'''')'
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ENDIF
	ENDIF  ! not overlay

	IF ( all_1_dep ) only_val = val1

* pass the data to PLOT+

	   WRITE ( ppl_buff, 3005 ) bad_val4, dep_ax
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	   IF (overlay) SOVER = .true.
	   icode = 0
	   npts2 = nmasked
	   IF (indep_nok.EQ.1 .OR. dep_nok.EQ.1) THEN
	      dep_dat(1) = dep_lo
	      indep_dat(1) = ind_min
	      npts2 = 1
	      indep_nok = 1
	      IF (line_symbol.EQ.qual_off .OR. 
     .            line_symbol.EQ.unspecified_int4) 
     .            line_symbol = qual_on  ! draw point with a symbol
	   ENDIF

	   
* set PLOT+ memory required, longitudes, latitudes, color var, gaps var

	   plot_mem_used = 2* 4* npts2

	   CALL PPLLDX_envelope(icode,indep_dat,dep_dat,npts2,
     .                             t1_date, tref, dt_min, plot_mem_used)

* ... increment number of lines on plot
	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1
	   CALL LINE_STYLE(line_symbol, sym_size, skipsym, color, color1, use_line,
     .			   do_dash, dashstyle, nline_in_mem, nline_on)

* * * * * * *    - - - LOAD DATA - - - * * * * * * * * *

* load color-by data into dep_dat

	ipl = nmv
	mv   = mv_list( ipl )
	cx   = cx_list( ipl )
	grid = cx_grid( cx )

* set up a color-by variable

	IF (cxdsg_empty_set) THEN
*          ... The dep_dat setup from above remains in place.
	   need_histo = .FALSE.
	   dep_min = -1.
	   dep_max =  1.
	   dep_dat(1) = 1.
	   dep_dat(2) = 1.
	   
	   CALL RIBBON_PLOTKEY_SETUP (overlay, status)
	ELSE


* ... For a dsg dataset, they can ask to color by a feature-level variable
*     such as an ID. That variable can be a string. Either way use it to 
*     create colorbar label strings and then set values to integer trajectory-
*     number for drawing the trajectories.

* determine 6D shape of context, get size of color var
	   CALL GET_CX_DIMS_ZERO( cx, ndim, dim )
	   dep_dim = dim(1)
	   npts2 = CGRID_SIZE(cx)

	   CALL EXTRACT_LINE (	cx,
     .				memry(mv)%ptr,
     .				mv,
     .				dep_dat,
     .				dep_dim,
     .				ndim,
     .				npts2,
     .				status	)
 
	   IF ( status .NE. ferr_ok ) GOTO 5000

* default color key
	   CALL RIBBON_PLOTKEY_SETUP (overlay, status)
	   IF (status .NE. ferr_ok) GOTO 5000

	   set_lev = .FALSE.  ! will auto-set levels

* If color by a nfeatures variable, expand to the length of the obs axis.
* If they didn't specify /LEVELS, set levels, use a centered-label color 
* key and label the key with the feature variable. Do this even for numeric 
* feature-id, label by feature variable, in order of feature number. These
* aren't taken to be increasing numeric values that could be used to set 
* numeric levels.

* Only do this for Trajetory data - for a dataset of type Points, number of points 
* is always number of features and features are length 1.

	   IF (npts2.EQ.nfeatures .AND. is_traj) THEN

	      IF (qual_given(slash_plot_levels) .EQ. 0) THEN
	         CALL SET_DSG_ID_LEV 
     .              (dset, cx, nfeatures, dep_dat, mr_type(mv), changed_key)
 
	        set_lev = .TRUE.
	      ELSE

* If they said /levels=v this doesnt make sense.  Just use linear levs.
	         IF (need_std) THEN
	            need_std = .FALSE.
	            CALL USE_LINEAR_LEVELS
	         ENDIF
	         set_lev = .TRUE.
	      ENDIF

	      IF (npts2 .EQ. nfeatures) THEN
	         DO i = 1, npts2
	            dep_dat(i) = i
	         ENDDO
	      ENDIF

* Expand feature number to fill the obs direction using the mask
	      CALL DSG_OBS_BY_FEATURE_VAR (dset, cx, nfeatures, obsdimlen, dep_dat)
	      npts2 = npts

* Set the cx_plot  limits from the command context so the plot labels will be correct
	      CALL DSG_ID_CX (cx, nfeatures)

	   ENDIF
	
	   IF (mask_applied) CALL PLOT_DSG_APPLY_OBS_MASK (dset, cx_list(nmv), 
     .		   nfeatures, obsdimlen, dep_dat, nmasked)
	
* Compute the mean and standard dev for ribbon-color variable, 
	   CALL MINMAX( dep_dat, nmasked, bad_val4, dep_min, dep_max, dep_nok )

* Compute the mean and standard dev for ribbon-color variable, 
* needed for computing color levels. Results stored in PPLUS common.
* If histogram-based levels are requested, compute the 
* histogram bins.
* send REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* If the user set some /LEVELS then compute_mnstd just returns 


	ENDIF ! cxdsg_empty_set


	IF (need_histo) THEN
* create temporary buffer to contain workspace
	   CALL CREATE_TEMP_MEM_VAR( cx, mvh_temp, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   plot_mem_used = plot_mem_used + 2* npts
	   CALL COMPUTE_HISTO_BINS (dep_dat, memry(mvh_temp)%ptr, 
     .            mr_bad_data(mv), nmasked, status) 
     
* ... clean up temporary variable
	   CALL DELETE_VARIABLE( mvh_temp )
	ELSE IF (.NOT.set_lev) THEN
	   rbad = bad_val4
	   CALL COMPUTE_MNSTD (dep_dat, mr_bad_data(mv), need_std, nmasked, rbad, status)

	   IF (need_std .AND. status.NE.ferr_ok) THEN
* ... set up for automatic levels
	      CALL PPLCMD ( from, line, 0, 'LEV,()', 1, 1 )
	      CALL USE_LINEAR_LEVELS
	      status = ferr_ok
	   ENDIF

	ENDIF
 
* pass the data to PLOT+

	WRITE ( ppl_buff, 3005 ) mr_bad_data( mv ), dep_ax
	
	
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	IF (overlay) SOVER = .true.

* See pplldx comments. When loading the color-by variable lon vertical time axis
* this forces the right combination of settings.

	icode = 2
	IF (indep_nok .EQ. 1) THEN
	   dep_dat(1) = dep_min
	   npts = 1
	ENDIF
	CALL PPLLDX_envelope(icode,indep_dat,dep_dat,nmasked,
     .                             t1_date, tref, dt_min, plot_mem_used)

* ... increment number of lines on plot
	nline_on     = nline_on     + 1
	nline_in_mem = nline_in_mem + 1

* ... assign line style for the data plotting:
	CALL LINE_STYLE(line_symbol, sym_size, skipsym, color, color1, use_line,
     .			   do_dash, dashstyle, nline_in_mem, nline_on)
 
*     Create a gaps-variable. Any masking is applied as the gaps are marked.

	IF (indep_nok .EQ. 1) THEN
	   dep_dat(1) = 1.
	   npts = 1
	ELSE
	   IF (.NOT.cxdsg_empty_set) CALL DSG_OBS_MARK_GAPS (dset, cx, nfeatures, obsdimlen, dep_dat)
	ENDIF
	
	CALL PPLLDX_envelope(icode, indep_dat, dep_dat, nmasked,
     .                             t1_date, tref, dt_min, plot_mem_used)

	nline_on     = nline_on     + 1
	nline_in_mem = nline_in_mem + 1

	ppl_buff = ' '
	gap_var = 4
	WRITE ( ppl_buff, 3007 ) gap_var
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1)
 3007	FORMAT ( 'GAPLOC ',I5 )

	CALL LINE_STYLE(line_symbol, sym_size, skipsym, color, color1, use_line,
     .			   do_dash, dashstyle, nline_in_mem, nline_on)

        nline_on = nline_on + color1 - 1

* axis scaling and formatting apply possible user settings
* ... independent axis

	IF ( .NOT.overlay ) THEN

* ... dependent axis scaling
*  force axis scaling if the data has no range

	   IF ( pdeplim .GT. 0 ) THEN
	      IF (pylim.GT.0 .AND. .NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF

	      CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(pdeplim):qual_end(pdeplim)),
     .			dep_dim, dep_min, dep_max, delta,
     .			formatted, range_rqd, cal_id_1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000

	      CALL AXIS_ENDS( dep_ax, the_taxis, grid, dep_min, dep_max, 
     .                        delta, indep_is_log, dep_axtyp, versus, status )
	      IF ( status .NE. ferr_OK ) THEN
                 first = dep_min
                 last  = dep_max
                 GOTO 5170
              ENDIF

	   ELSEIF ( all_1_dep ) THEN
	      IF (only_val .EQ. bad_val4) val1 = 0.0	! 10/99
	      delta = 1.
	      IF (val1 .NE. 0) delta = 0.1* val1
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid,val1-delta,val1+delta,delta,
     .                        indep_is_log, dep_axtyp, versus, status )   
	      IF ( status .NE. ferr_OK ) THEN
                  first = val1-delta
		  last  = val1+delta
		  GOTO 5170
	      ENDIF   

	   ENDIF

           ax1 = axdir(indep_dim)
  	   CALL PPLCMD ( from, line, 0, 'SET AX_HORIZ '//ax1, 1, 1 )
	ENDIF

*  TITLES
* Main plot title. The main plot label is the the ribbon color variable
*
	hlen = xlen ! single-precision xlen from PPLUS common -> double prec. var

	IF ( .NOT.no_labels ) 
     .          CALL LINE_PLOT_LABELS (nmv, nkey_entries, ndv, cx_list, 
     .		this_no_range, overlay, versus, nokey, .FALSE., 
     .		tstyle, cal_id_1, ribbon_var, indep_lab, dep_lab, 
     .		ind_min, dep_len, hlen, nfeatures, dset, 
     .		dsg_feature_var(dset) )

* set flag indicating a 2D plot is on the screen, and now that things are set up,
* tell any subsequent plots that it's an XY plot
	IF (.NOT.overlay) twodee_on = .TRUE.

	plot_axis(1) = 1
	plot_axis(2) = 2

* successful completion

	status = ferr_ok
	IF (overlay) iautot = iautot_save
	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
 	IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
	RETURN

 5110	dep_ax = LEFINT( ndim, slen )
	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'specified data is not a line'//pCR//
     .			' - its a '//dep_ax(:slen)//'D region: "'//
     .			cmnd_buff(:len_cmnd)//'"', *5000 )
 5120	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'overlay is on a different axis'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )

 5130	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'longitude,latitude coordinate variables not found'//pCR//
     .			'Is this a trajectory dataset?'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
        
 5140	CALL ERRMSG( ferr_dim_underspec, status, 
     .			'longitude,latitude coordinates must be on observation-axis'//pCR//
     .			'Is this a trajectory dataset?'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
     
        
 5150	CALL ERRMSG( ferr_data_type, status, 
     .			'Cannot plot data type of string on the observation axis'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
        

 5170	buff1 = LEFT_REAL (first, '(G15.3)', slen)
	buff2 = LEFT_REAL (last, '(G15.3)', slen2)
	CALL ERRMSG( ferr_out_of_range, status,
     . 'Limits for log axis negative or too small: '//
     .  buff1(:slen)// ' : '//  buff2(:slen2), *5000 )

 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'Data grid is not a DSG grid ',
     .		*5000 )

	END
