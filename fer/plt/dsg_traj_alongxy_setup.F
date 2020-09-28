        SUBROUTINE DSG_TRAJ_ALONGXY_SETUP ( dset,
     .				overlay, no_labels, symbol, sym_size, color, 
     .				color1, use_line, do_dash, dashstyle, nokey,
     .				only_val, skipsym, mv, cx, indep_dat, dep_dat, status )

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
* based on dsg_traj_plot_set_up, Set up to draw trajectory plots from a dsg trajectoryProfile dataset
* V762 *acm* 9/20 
* v763 *acm* 9/20 Plots of id-variables in trajectory-profile, timeseries-profile data
* V763 *acm* 9/20 Also plot/along=xy for timeseries or profile data to make a map.

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
#include "tmap_dset.parm"

* calling argument declarations:
	LOGICAL		overlay, no_labels, use_line,
     .			nokey
	INTEGER		dset, symbol, color, color1, do_dash, skipsym,
     .			mv, cx, status
	REAL		sym_size, dashstyle(*), only_val
	REAL		indep_dat(*), dep_dat(*)     ! dynamic space allocation
 
* V500 *kob* 3/99- up VAR_CODE and VAR_UNITS to 64 chars
* internal variable declarations:

	LOGICAL		NO_LINE_RANGE, TM_ITSA_DSG_RAGGED,
     .			versus, all_1_ind, all_1_dep, formatted, use_nice, 
     .			this_no_range(max_line_on_plot), indep_is_log, set_lev, 
     .			its_dsg, its_cmpnd, mask_applied, user_hlim, is_dsg, 
     .			is_cmpnd, dirs(4), tsprof, its_traj, dsg_as_time, 
     .			dsg_as_traj, as_traj, color_by_station
     .			
	CHARACTER	LEFINT*8, LEFT_REAL*16, FULL_VAR_TITLE*2040,
     .			indep_ax*1, dep_ax*1, tstyle*3, 
     .			buff1*40, buff2*16, t1_date*14, tref*14, fmt*4
	INTEGER		TM_LENSTR1, CGRID_SIZE, CGRID_AXIS, DSG_WHATS_IT,
     .			TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, TM_DSG_OBS_GRID,
     .			slen, ito, indep_lab, dep_lab, nline_in_mem,
     .			indep_dim, npts, dep_dim, npts2, ndv, var1,
     .			grid, ipl, ndim, dims(6),
     .			slen2, slen3, tax, plot_mem_used, i,
     .			pxlim, pylim, phlim, pvlim, pindeplim, pdeplim,
     .			dep_axtyp, indep_axtyp, the_taxis, ribbon_var, 
     .			gap_var, iiunits, iaxis, hblk1, mvh_temp, 
     .			nkey_entries, itmp, igrd_save, cal_id_1,
     .			icode, dset_dsg, nfeatures, obsdimlen, fline,
     .			idim, coord_lm, ivar, dim(nferdims), numcx,
     .			dep_nok, indep_nok, line_symbol, nmasked, nftrsets, 
     .			row_size_lm, orientation, ifeature, imask, iftr, 
     .			lm_index, base, iline, ic, irow_f, irow_l, nobs, 
     .			irow_start, irow_end


        INTEGER         iunits, junits
	CHARACTER*1     axdir(nferdims), ax1

	REAL		val1, dt_min, lo, hi, dep_len, 
     .			ind_len, first, last, hlen, dep_lo, 
     .			dep_hi, indep_lo, indep_hi, pad
	REAL		ind_min, ind_max, dep_min, dep_max, 
     .			delta, tmp, labsize, bad, psum, pcnt
	REAL*4 rbad

* local parameter declarations:
	LOGICAL		range_rqd
	PARAMETER     ( range_rqd = .TRUE. )
	DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

        INTEGER, DIMENSION(:), ALLOCATABLE :: cmpnd_index
        REAL, DIMENSION(:), ALLOCATABLE :: coord
        LOGICAL, DIMENSION(:), ALLOCATABLE :: process_feature
        LOGICAL, DIMENSION(:), ALLOCATABLE :: process_obs

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'setting up plot', 15)

* initialize

	ind_max = arbitrary_small_val8
	ind_min = arbitrary_large_val8
	versus = .TRUE.
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
        nkey_entries = 1
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

	its_dsg = .TRUE.    
	grid = cx_grid(cx)
	color_by_station = DSG_WHATS_IT(grid) .EQ. pdsg_fs_dim

	grid = dsg_xlate_grid(dset)

	CALL TM_DSG_FACTS( grid, idim, obsdimlen, fline, its_dsg, its_cmpnd )

	IF (its_cmpnd) THEN
	   CALL TM_DSG_FTRSET_FACTS( grid, dset, orientation, 
     .                  nfeatures, nftrsets, is_dsg, is_cmpnd, status )
	ELSE   
	   nftrsets = 0
	   nfeatures = TM_DSG_NFEATURES( grid )
	   orientation = dsg_orientation(dset)
	ENDIF

	tsprof = (orientation .EQ. pfeatureType_TimeseriesProfile)
     
	CALL DSG_PLTALONG_SETUP (dset, idim, its_traj, dsg_as_traj,  
     .              dsg_as_time, status)
	as_traj = orientation.EQ.pfeatureType_TrajectoryProfile .AND. idim.EQ.x_dim


	ul_dolab(z_dim) = .TRUE.

	ALLOCATE (process_obs(obsdimlen )) 
	ALLOCATE (process_feature(nfeatures )) 
        ALLOCATE (cmpnd_index(nfeatures))
        ALLOCATE (coord(nfeatures))

* check for improper data supplied
	IF ( overlay ) THEN
	   CALL GET_CX_DIMS( cx, ndim, dims )
           IF ( twodee_on ) indep_dim = dims(1) 
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


* This will cause the x-y context limits to be set up
	DO idim = 1, 4
	   dirs(idim) = .TRUE.
	ENDDO
	dirs(z_dim) = .FALSE.
	CALL MAKE_DSG_FEATURE_MASK_DIRS(dset, cx, process_feature, nfeatures, dirs)
     
	IF (is_cmpnd) THEN
	   lm_index = dsg_loaded_lm(dsg_index_var(dset))

	   DO i = 1, nfeatures
	      cmpnd_index(i) = dsg_linemem(lm_index)%ptr(i) + 1
	   ENDDO
	ELSE
	   DO i = 1, nfeatures
	      cmpnd_index(i) = i
	   ENDDO
	ENDIF
	   
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))


* * * * * * *    - - - INDEPENDENT AXIS DATA - - - * * * * * * * * *
	
* Get the LONGITUDE coordinate data

	idim = 1
	iaxis = grid_line(idim, dsg_xlate_grid(dset))
	iunits = line_unit_code ( iaxis )
	IF (iunits .NE. 4) GOTO 5130  ! must be longitude coordinates

	ivar = dsg_coord_var(idim,dset)
	coord_lm = dsg_loaded_lm(ivar)  ! line memory table indices

	DO iftr = 1, nfeatures
	   IF (tsprof) THEN
	      coord(iftr) = dsg_linemem(coord_lm)%ptr(cmpnd_index(iftr))
	   ELSE
	      coord(iftr) = dsg_linemem(coord_lm)%ptr(iftr)
	   ENDIF
	ENDDO

	npts = nfeatures	

*  Check for longitudes crossing the dateline. Looks only at location data in features
*  included in any feature-level mask, which is applied in PLOT_DSG_APPLY_OBS_MASK

	user_hlim = pindeplim .GT. 0
c	CALL PLOT_DSG_CHECK_LON360 (dset, cx, indep_dat, bad_val4, 
c     .                    npts, nfeatures, ind_min, ind_max, user_hlim)

* Note process_fteset mask masks out only the trajectories masked out by a traj-mask
* not the region spec

	nmasked = nfeatures
	imask = 0

	DO i = 1, nfeatures
	   IF (process_feature(i)) THEN
	      imask = imask+1
	      indep_dat(imask) = coord(i)
	   ENDIF
	ENDDO

	nmasked = imask

	IF (nmasked .EQ. 0) THEN
	   cxdsg_empty_set = .TRUE.

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
	   mask_applied = nmasked .LT. nfeatures
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

	   IF (.NOT. all_1_ind) THEN
	      pad = (ind_max - ind_min)* 0.01
	      ind_min = ind_min - pad
	      ind_max = ind_max + pad
	   ENDIF

           CALL AXIS_ENDS(indep_ax,indep_dim,grid,ind_min,ind_max,
     .                     delta, indep_is_log, indep_axtyp, versus, status)
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

	   ivar = dsg_coord_var(idim,dset)
	   coord_lm = dsg_loaded_lm(ivar)  ! line memory table indices

	   DO iftr = 1, nfeatures
	      IF (tsprof) THEN
	         coord(iftr) = dsg_linemem(coord_lm)%ptr(cmpnd_index(iftr))
	      ELSE
	         coord(iftr) = dsg_linemem(coord_lm)%ptr(iftr)
	      ENDIF
	   ENDDO

	   all_1_dep = .TRUE.
	   only_val  = bad_val4

	   imask = 0
	   DO i = 1, nfeatures
	      IF (.NOT.process_feature(i)) CYCLE
 	      imask = imask + 1
	      dep_dat(imask) = coord(i)
	   ENDDO

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
	   IF (.NOT.all_1_dep .AND. dep_nok.NE.0) THEN
	      pad = (dep_max - dep_min)* 0.01
	      dep_min = dep_min - pad
	      dep_max = dep_max + pad
	   ENDIF

	ENDIF ! cxdsg_empty_set


	IF (.NOT.overlay) THEN

	   delta = unspecified_val8    
	   CALL AXIS_ENDS(dep_ax, dep_dim, grid, dep_min, dep_max,
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


* ... For a dsg dataset, they can ask to color by an observed variable, or a traj-level
*     variablesuch as an ID. That variable can be a string. Either way use it to 
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
 
	IF (color_by_station) THEN

* Set up to label the plot key 
	   CALL SET_DSG_ID_LEV 
     .              (dset, cx, nftrsets, dep_dat, mr_type(mv), changed_key)
	   set_lev = .TRUE.

* Set the cx_plot  limits from the command context so the plot labels will be correct
	   CALL DSG_ID_CX (cx, nftrsets)
	      
* Put the trajectory-number into dep_dat

	   DO i = 1, nfeatures
	      dep_dat(i) = cmpnd_index(i)
	   ENDDO

	   npts2 = nfeatures

	ENDIF

	IF ( status .NE. ferr_ok ) GOTO 5000

* Issue a note about averaging to color-by one value per profile

	IF (dsg_as_traj .AND. npts2.EQ.obsdimlen) THEN

	   IF (cx_lo_ww(t_dim, cx_cmnd).EQ.unspecified_val8 .OR. 
     .         cx_lo_ww(t_dim, cx_cmnd).NE.cx_hi_ww(t_dim, cx_cmnd)) THEN


	      IF (orientation.EQ.t_dim) THEN
		    CALL WARN (
     .            'PLOT/ALONG= with /T=LO:HI colors the plot with AVE of timeseries data in that range')
	      ELSE
		    CALL WARN (
     .            'PLOT/ALONG= with /Z=LO:HI colors the plot with AVE of profile data in that range')
	      ENDIF

	   ENDIF

	ENDIF

* if color-by trajectory-id variable or timeseries-station variable,
* use the index variable 


* default color key
	CALL RIBBON_PLOTKEY_SETUP (overlay, status)
	IF (status .NE. ferr_ok) GOTO 5000

	set_lev = .FALSE.  ! will auto-set levels
	bad = mr_bad_data(mv)

* ... Now get the data values to use in the color-by-variable plot along
*     the trajectories.

        IF (npts2 .EQ. nfeatures) THEN


	   imask = 0
	   DO ifeature = 1, nfeatures 

	      IF (.NOT.process_feature(ifeature)) CYCLE

	      imask = imask + 1
	      dep_dat(imask) = dep_dat(ifeature) 

	   ENDDO

	ENDIF  

        IF (npts2 .EQ. obsdimlen) THEN

           iline = 0
	   irow_f = 1
	   irow_l = 0
	   imask = 0

	   DO ifeature = 1, nfeatures 
	      nobs = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	      IF (nobs .EQ. 0) GOTO 222
	      irow_l = irow_f + nobs - 1
	      IF (.NOT.process_feature(ifeature)) GOTO 222
	      iline = iline + 1

* ... get observation-level mask for this feature

             base = irow_f - 1
	     CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base,
     .				  process_obs(irow_f), nobs)

* Find first and last points for this feature applying the user's context 
* This just masks the range of the dependent variable. Still need to apply the 
* process_obs mask to the dependent variable when plotting.

	      irow_start = 0
	      irow_end = 0

	      DO ic = irow_f, irow_l
	         IF (.NOT.process_obs(ic) .AND. irow_start.EQ. 0) THEN
	            CYCLE
	         ELSE 
	            irow_start = ic
		    EXIT ! from loop
                 ENDIF
	      ENDDO

	      DO ic = irow_l, irow_f, -1
	         IF (.NOT.process_obs(ic) .AND. irow_end.EQ. 0) THEN
	            CYCLE
	         ELSE 
	            irow_end = ic
		    EXIT ! from loop
                 ENDIF
	      ENDDO

	      imask = imask + 1
	      IF (irow_start.EQ.0 .OR. irow_end.EQ.0) THEN
	         dep_dat(imask) = bad_val4
	      ELSE
	         
		 psum = 0.
		 pcnt = 0.
		 DO i = irow_start, irow_end

		    IF (dep_dat(i) .NE. bad) THEN
		       psum = psum + dep_dat(i) 
		       pcnt = pcnt + 1.
		    ENDIF

	         ENDDO
		 
		 dep_dat(imask) = bad_val4
		 IF (psum .GT. 0.) dep_dat(imask) = psum / pcnt
	      ENDIF

* ... prepare for the next feature

 222	      irow_f = irow_l + 1

	   ENDDO

	ENDIF  ! npts2 .EQ. obsdimlen


		
* Compute the mean and standard dev for ribbon-color variable, 
	CALL MINMAX( dep_dat, nmasked, bad_val4, dep_min, dep_max, dep_nok )

* Compute the mean and standard dev for ribbon-color variable, 
* needed for computing color levels. Results stored in PPLUS common.
* If histogram-based levels are requested, compute the 
* histogram bins.
* send REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* If the user set some /LEVELS then compute_mnstd just returns 

	IF (need_histo) THEN
* create temporary buffer to contain workspace
	   CALL CREATE_TEMP_MEM_VAR( cx, mvh_temp, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   plot_mem_used = plot_mem_used + 2* nmasked
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

	ENDIF ! cxdsg_empty_set
c----------------------

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
 
*     Create a gaps-variable. Any masking is applied as the gaps are marked.

	IF (as_traj) THEN 

* first un-masked trajectory, is trajectory # ic

	   irow_end = 0
	   DO i = 1, nfeatures
	      IF (process_feature(i) ) THEN
	         ic = cmpnd_index(i)
		 EXIT
	      ENDIF
	   ENDDO

	   imask = 0
	   DO i = 1, nfeatures
	      imask = imask + 1
	      dep_dat(imask) = 0.
	      IF (cmpnd_index(i).NE.ic .AND. imask.GT.1) THEN
	         dep_dat(imask-1) = 1.
		 ic = cmpnd_index(i)
              ENDIF
	   ENDDO

	
	   CALL PPLLDX_envelope(icode, indep_dat, dep_dat, nmasked,
     .                             t1_date, tref, dt_min, plot_mem_used)

	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1

	   ppl_buff = ' '
	   gap_var = 4
	   WRITE ( ppl_buff, 3007 ) gap_var
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1)
 3007	   FORMAT ( 'GAPLOC ',I5 )

	ENDIF

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

	   ELSEIF ( all_1_dep ) THEN
	      IF (only_val .EQ. bad_val4) val1 = 0.0	! 10/99
	      delta = 1.
	      IF (val1 .NE. 0) delta = 0.1* val1
	      
	      CALL AXIS_ENDS( dep_ax, the_taxis, grid,val1-delta,val1+delta,delta,
     .                        indep_is_log, dep_axtyp, versus, status )   

	   ENDIF

           ax1 = axdir(indep_dim)
  	   CALL PPLCMD ( from, line, 0, 'SET AX_HORIZ '//ax1, 1, 1 )
	ENDIF

*  TITLES
* Main plot title. The main plot label is the the ribbon color variable
*
	hlen = xlen ! single-precision xlen from PPLUS common -> double prec. var

	IF ( .NOT.no_labels ) 
     .          CALL LINE_PLOT_LABELS (1, nkey_entries, ndv, cx, 
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

	
* not trajectory after all, will just do a ribbon-plot.

 4000	CONTINUE

	DEALLOCATE (process_obs)
	DEALLOCATE (cmpnd_index) 
	DEALLOCATE (process_feature)
	DEALLOCATE (coord)

* no need to unwind things at the end of xeq_plot.
	IF (color_by_station) grid_is_dsg = .FALSE. 


	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
 	IF (itmp .NE. mnormal) CALL TM_DEALLO_DYN_LINE( itmp )
	GOTO 4000

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
                
 5150	CALL ERRMSG( ferr_data_type, status, 
     .			'Cannot plot data type of string on the observation axis'//pCR//
     .			'"'//cmnd_buff(:len_cmnd)//'"', *5000 )
        
 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'Data grid is not a DSG grid ',
     .		*5000 )

	END
