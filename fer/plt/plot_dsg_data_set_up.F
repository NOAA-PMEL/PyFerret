	SUBROUTINE PLOT_DSG_DATA_SET_UP( nfeatures, nobs_total, 
     .					 dsg_as_time, status )

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
* modify the interpretation of the command data so it looks like
* a collection of 1D lines in the "feature-type" direction
* this will be accomplished by generating a list of contexts, each for a
* single line, each pointing to the same, shared memory-resident array
* Ideas similar to PLOT/ALONG and LIST command for DSG data.

* programmer - Ansley Manke
*
* V7.3+ *acm* 12/2017
* V7.6  *acm* 11/2019  Allow for the possibility of 0 obs in a feature.
*                      If this is the case, skip it.
* V760 *acm* 3/20      Set up for case when constraints result in no features
* V763 *acm* 9/20      Issue a note about data taken from z profiles for xy plots.

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'command.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xplot_setup.cmn'
	include 'xtm_grid.cmn_text'
	include 'xtext_info.cmn'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xdsg_context.cmn'
#include "tmap_dset.parm"

	
* calling argument declarations:
	LOGICAL dsg_as_time
	INTEGER	nfeatures, nobs_total, status

* internal variable declarations:
	LOGICAL process_feature(nfeatures), process_obs(nobs_total), 
     .		relevant_coord(4), has_keys, its_dsg, its_cmpnd,
     .		multi_var, new_line, first_warn
	INTEGER	CGRID_SIZE, DSG_WHATS_IT, TM_DSG_DSET_FROM_GRID,
     .		iline, nlines, idim, ndim, the_dim, max_feature_lines, cx0, 
     .		the_cx, slen, tlen, cx, grid, feature_line, orientation, 
     .		obsdimlen, dset, gxlate, igrid, ifeature, coord_lm(4), 
     .		ivar, row_size_lm, nobs, irow_f, irow_l, iaxis, ic, 
     .		irow_start, irow_end, base, icxvar, lm_index, 
     .		itim_f, ista, ntim, mv, dep_dim, npts2
     
        INTEGER*8 rqst_n
	REAL	value, val_last, val_edge, vdel, bad
	CHARACTER buff1, LEFINT*8, buff8*8,  buff*256

        INTEGER, DIMENSION(:), ALLOCATABLE :: station_index
        REAL, DIMENSION(:), ALLOCATABLE :: dep_dat

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'set up line plots', 15)
* initialize
	cx = is_cx(1)
	mv = is_mr(1)
	grid = cx_grid(cx)
	dset = TM_DSG_DSET_FROM_GRID( grid )
	first_warn = .TRUE.

* get the plot orientation 4=timeseries,3=profile,1=trajectory
* for 1-D plots, plotting trajectory on an unraveled axis 
* (possibly point, which is feature type 0, would be treated as a 1 here? Collection of points)
* Trajectories as plot/vs/ribbon plots are set up elsewhere.

	its_dsg = .TRUE.
	CALL TM_DSG_FACTS( grid, orientation, obsdimlen, feature_line, 
     .			   its_dsg, its_cmpnd )
	IF (.NOT. its_dsg) GOTO 5500
	gxlate = dsg_xlate_grid(dset)
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

* ... unrecognized orientation	         
	IF (orientation .EQ. 0) GOTO 5200  ! ??

* By default, if it's trajectoryProfile or timeseriesProfile plot as profiles
* as if plot/along=z. (If it's trajectoryProfile plotted as trajectory, 
* plot_dsg_data_set_up is not called.)

	IF (orientation .EQ. pfeatureType_TimeseriesProfile ) orientation = pFeatureType_Profile
	IF (orientation .EQ. pfeatureType_TrajectoryProfile ) orientation = pFeatureType_Profile

* If its an instance var, this can be plotted as an ordinary line plot in E.

	IF (DSG_WHATS_IT(grid) .EQ. pdsg_instance .OR.  CGRID_SIZE(cx) .EQ. 1) THEN
	   grid_is_dsg = .FALSE.
	   GOTO 5000
	ENDIF

* What coordinate variables are represented in the grid?

	DO idim = 1, 4
	   iline = grid_line(idim, gxlate)
	   relevant_coord(idim) =  (iline .NE. mnormal)
	   IF (relevant_coord(idim)) THEN
	      ivar = dsg_coord_var(idim,dset)
	      coord_lm(idim) = dsg_loaded_lm(ivar)  ! line memory table indices
	   ENDIF
	ENDDO

* create a mask showing which features to plot

	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)


	IF (dsg_as_time) THEN

* IF it's PLOT/ALONG=XY/Z=/LO:HI or PLOT/ALONG=T/Z=/LO:HI then the locgic here returns 
* the first value encountered within the Z range along each profile.
* IF no /Z= region is given then it returns the first value found along each profile.
* Issue a note about this.

	   IF (cx_lo_ww(z_dim, cx_cmnd) .EQ. unspecified_val8  ) THEN
	      CALL WARN (
     .        'PLOT/ALONG=  with no /Z= specified uses the first data value in each profile')
	      first_warn = .FALSE.
	   ENDIF

* Get the station-index variable for trajectoryProfile or timeseriesProfile data
* Change the grid we are using to the grid of the station-index variable.


	   ALLOCATE (station_index(nfeatures))
	   lm_index = dsg_loaded_lm(dsg_index_var(dset))
	   DO ifeature = 1, nfeatures
	      station_index(ifeature) = dsg_linemem(lm_index)%ptr(ifeature) + 1
	   ENDDO

* Get the observations, will select the one per profile to put into the timeseries
* for each station.

	   ALLOCATE (dep_dat(nobs_total))
	   CALL EXTRACT_LINE ( cx,
     .				memry(mv)%ptr,
     .				mv,
     .				dep_dat,
     .				dep_dim,
     .				ndim,
     .				npts2,
     .				status	)
	   ! compare nobs_total and npts2.  ndim should be 1
	   bad = mr_bad_data(mv)	   

* dynamic memory which will contain the data on each feature selected for the timeseries
	
	   rqst_n = nfeatures ! I*8 argument
	   CALL TM_GET_LM_DYNMEM(rqst_n, cxdsg_lm_tsdat, status)

	   grid = gxlate

	ENDIF


* Reset nlines: how many stations in the request?  A line will be the data at
* one level of the profile, becoming a timeseries at the station.

	nlines = 0
	ista = 0
	ntim = 0
	DO ifeature = 1, nfeatures 
	   IF (.NOT.process_feature(ifeature)) CYCLE

	   nobs = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	   IF (nobs .GT. 0) THEN

	      IF (dsg_as_time) THEN

	         ntim = ntim + 1
		 IF (station_index(ifeature) .NE. ista) THEN
                    nlines = nlines+1
                    ista = station_index(ifeature)
	         ENDIF

* Initialize nfeatures values on timeseries station-line
	         CALL PUT_LINE_COORD ( dsg_linemem(cxdsg_lm_tsdat)%ptr, ntim, bad )

	      ELSE

	         nlines = nlines+1

	      ENDIF

	   ENDIF ! nobs
              
	ENDDO


* Multi-feature DSG data can only plot one variable

	multi_var = .FALSE.
	IF (num_uvars_in_cmnd .GT.1) THEN 
	   IF (nlines .GT. 1) GOTO 5100
	   multi_var = .TRUE.
	ENDIF

* ... If the mask or constraints results in no data selected, load nominal 
*     data so a blank plot can be drawn.

	IF (nlines .EQ. 0) THEN
	
           nobs = 2
           nlines = 1
           iline = 1
           the_cx = is_cx(1)

           is_mr(iline) = is_mr(1)  ! a single mr shared by all

* Define a new set of coordinates, the lo and hi in the current context

           igrid = unspecified_int4
           CALL TM_ALLO_TMP_GRID (igrid, status)
           IF (status .NE. ferr_ok) GOTO 5000

           CALL TM_COPY_GRID_W_LINE_USE (grid, igrid)
           CALL TM_DEALLO_DYN_LINE( grid_line(orientation, grid) )

           CALL TM_ALLO_TMP_LINE(iaxis, status)
           CALL GET_LINE_DYNMEM (nobs, iaxis, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	   
           grid_line(orientation, igrid) = iaxis
           cx_grid(the_cx) = igrid
           CALL TM_USE_LINE(iaxis)

* Inherit axis properties
           CALL TM_COPY_LINE (grid_line(orientation,grid), iaxis)
           line_regular(iaxis) = .FALSE.

* rename the axis. 
           CALL TM_NEW_LINE_NAME ( line_name(iaxis), buff )
           line_name(iaxis) = buff

           irow_start = 1
           irow_end   = 2

           ic = irow_start
           CALL PUT_LINE_COORD(linemem(iaxis)%ptr, ic, cxdsg_constrain_lo(orientation))

           ic = irow_end
           CALL PUT_LINE_COORD(linemem(iaxis)%ptr, ic, cxdsg_constrain_hi(orientation))

           cx_feature(the_cx) = 1

           DO idim = 1, t_dim
              cx_lo_ss(the_cx,idim) = unspecified_int4
              cx_hi_ss(the_cx,idim) = unspecified_int4
           ENDDO

* ... set up the feature-axis positions 
           cx_by_ss(orientation, the_cx) = .TRUE.
           cx_lo_ss(the_cx,orientation) = irow_start
           cx_hi_ss(the_cx,orientation) = irow_end

           cxdsg_empty_set = .TRUE.
           cxdsg_orient = dsg_orientation(dset)  ! for making label
	   
	ENDIF  ! nlines .EQ. 0

        IF (cxdsg_empty_set) GOTO 333

* have plot keys been requested?
	has_keys = qual_given(slash_plot_nokey) .EQ. 0
     .       .AND. qual_given(slash_plot_title) .EQ. 0

* how many individual lines are allowed
* (max_key_entries smaller than max_line_on_plot)
	max_feature_lines = max_noframe_context - is_cx(1) + 1
	IF (max_feature_lines .GT. max_line_on_plot) max_feature_lines =
     .						   max_line_on_plot

c A message is written in PLOT_SET_UP
	IF (has_keys .AND. nlines.GT.max_key_entries) has_keys = .FALSE.
	
	IF (nlines .GT. max_feature_lines) THEN
	   buff8 = LEFINT( nlines,  slen  )
	   CALL WARN('Too many Feature lines. Request would plot '
     .			// buff8(:slen) )
	   buff8 = LEFINT( max_feature_lines,  slen  )
	   CALL WARN('Plotting first '// buff8(:slen) )
	   nlines = max_feature_lines
	ENDIF


* populate the contexts

	CALL TRANSFER_CONTEXT( is_cx(1), cx_buff )

* For each station, populate a line in time using the first value
* from the profile at that station and time.

	the_dim = orientation
	IF (dsg_as_time) the_dim = t_dim

	cx0 = is_cx(1) - 1
        iline = 0
	irow_f = 1
	irow_l = 0

	IF (dsg_as_time) THEN
	   ista = 0
	   ntim = 0
	   itim_f = 1
	   the_dim = t_dim
	ENDIF

	the_cx = cx0

	DO ifeature = 1, nfeatures 
	   nobs = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	   IF (nobs .EQ. 0)  GOTO 222
	   irow_l = irow_f + nobs - 1
	   IF (.NOT.process_feature(ifeature)) GOTO 222

* ... get observation-level mask for this feature

           base = irow_f - 1
	   CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base,
     .				  process_obs(irow_f), nobs)

	   IF (dsg_as_time) THEN  ! new station?

	      new_line = .FALSE.
	      IF (station_index(ifeature) .NE. ista) THEN

* ... set the context of the previous station-line

	         IF (ista .GE. 1 .OR. ifeature.EQ.nfeatures) THEN
		    cx_feature(the_cx) = ista  ! ??
		    cx_by_ss(the_dim, the_cx) = .TRUE.
		    cx_lo_ss(the_cx,the_dim) = itim_f
		    cx_hi_ss(the_cx,the_dim) = ntim

		    itim_f = ntim+1

	         ENDIF

	         new_line = .TRUE.
	         iline = iline + 1
	         ista = station_index(ifeature)
	         the_cx = cx0 + iline
	         is_cx(iline) = the_cx 
	         is_mr(iline) = is_mr(1)  ! a single mr shared by all

* we are storing the data for station timeseries in temporary dynamic line
* storage so no need to think about mr()

		 CALL TRANSFER_CONTEXT( cx_buff, the_cx )

		 cx_lo_ss(the_cx,orientation) = unspecified_int4
		 cx_hi_ss(the_cx,orientation) = unspecified_int4

	      ENDIF

	   ELSE 

* Not dsg_as_time

	      iline = iline + 1
	      is_mr(iline) = is_mr(1)  ! a single mr shared by all
	   
	      the_cx = cx0 + iline
	      is_cx(iline) = the_cx
	      CALL TRANSFER_CONTEXT( cx_buff, the_cx )
	      new_line = .TRUE.

	   ENDIF ! IF dsg_as_time


* Define a new set of coordinates, as if an axis of the original total length,
* but the coordinates at indices irow_f through irow_l will contain the coordinates 
* from the feature that we want. Define a grid to contain this axis.

	   IF (new_line) THEN

	      igrid = unspecified_int4
	      CALL TM_ALLO_TMP_GRID (igrid, status)
	      IF (status .NE. ferr_ok) GOTO 5000

c	      IF (dsg_as_time) THEN  ! just get the time axis into new grid
c	         CALL TM_COPY_GRID (grid, igrid)
c		 DO idim = 1, nferdims
c		    IF (idim .NE. the_dim) grid_line(idim, igrid) = 0
c		 ENDDO
c	      ELSE
	         CALL TM_COPY_GRID_W_LINE_USE (grid, igrid)
	         CALL TM_DEALLO_DYN_LINE( grid_line(the_dim, grid) )
c	      ENDIF

	      CALL TM_ALLO_TMP_LINE(iaxis, status)
	      CALL GET_LINE_DYNMEM (nobs_total, iaxis, status)
	      IF (status .NE. ferr_ok) GOTO 5000
	   
	      grid_line(the_dim, igrid) = iaxis
	      cx_grid(the_cx) = igrid
	      CALL TM_USE_LINE(iaxis)
	      
* Inherit axis properties
	      CALL TM_COPY_LINE (grid_line(the_dim,grid), iaxis)
	      line_regular(iaxis) = .FALSE.

* rename the axis. (Is this necessary? Nice for debugging anyway)
	      CALL TM_NEW_LINE_NAME ( line_name(iaxis), buff )
	      line_name(iaxis) = buff

	   ENDIF  ! new_line

* Find first and last points for this feature applying the user's context 
* This just masks the range of the independent variable. Still need to apply the 
* process_obs mask to the independent variable when plotting.

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

	   IF (irow_start.EQ.0 .OR. irow_end.EQ.0) CYCLE  ! shouldnt happen.

	   IF (dsg_as_time) THEN

* When collecting points at a station for a timeseries of timeseriesProfile data, 
* choose the first point if more than one returned by the masking.  Store the coordinate
* value in linemem(iaxis) and the corresponding data from the profile in dsg_linemem(lm_tsdat)
* The time coordinates are in the n-features dimension, so get the time coordinate from 
* the current ifeature.

	      ntim = ntim + 1
	      value = dsg_linemem(coord_lm(the_dim))%ptr(ifeature)
	      CALL PUT_LINE_COORD (linemem(iaxis)%ptr, ntim, value)
	      CALL PUT_LINE_COORD (dsg_linemem(cxdsg_lm_tsdat)%ptr, ntim, dep_dat(irow_start) )  
	      IF (irow_start .NE. irow_end .AND. first_warn) THEN
	         CALL WARN (
     .            'PLOT/ALONG= /Z=LO:HI uses the first data value in the Z- range on each profile')
		 first_warn = .FALSE.
	      ENDIF
	   ELSE

* Put the coordinates from this feature-line into the locations
* on the orientation-axis corresponding to our index range.

	      DO ic = irow_start, irow_end
	         value = dsg_linemem(coord_lm(the_dim))%ptr(ic)
	         CALL PUT_LINE_COORD(linemem(iaxis)%ptr, ic, value)
	         IF (ic .GT. irow_start) THEN
	            vdel = (value-val_last)/2.
		    val_edge = value - vdel
		    CALL PUT_LINE_COORD(lineedg(iaxis)%ptr, ic, val_edge)
	         ENDIF
	         val_last = value
	      ENDDO
	      IF (ic .GT. irow_start+1) THEN 
	         val_edge = value + vdel
	         CALL PUT_LINE_COORD(lineedg(iaxis)%ptr, ic, val_edge)
	     ENDIF

*** To be able to recover the feature number, so keys can be labeled?

	      cx_feature(the_cx) = ifeature
	   
* ... set up the feature-axis positions 
	      cx_by_ss(the_dim, the_cx) = .TRUE.
	      cx_lo_ss(the_cx,the_dim) = irow_start
	      cx_hi_ss(the_cx,the_dim) = irow_end

	   ENDIF  ! dsg_as_time

* ... prepare for the next feature

 222	   irow_f = irow_l + 1

	   IF (iline .GE. max_feature_lines) EXIT
	ENDDO

* context for last station
	IF (dsg_as_time) THEN
	   
	   cx_feature(the_cx) = ista  
	   cx_by_ss(the_dim, the_cx) = .TRUE.
	   cx_lo_ss(the_cx,the_dim) = itim_f
	   cx_hi_ss(the_cx,the_dim) = ntim

	   cx_lo_ss(the_cx,orientation) = unspecified_int4
	   cx_hi_ss(the_cx,orientation) = unspecified_int4
	ENDIF

  333	CONTINUE

* make the rest of Ferret believe that all these lines were asked for
	IF (.NOT. multi_var) THEN
	   num_uvars_in_cmnd = nlines
	ELSE

* multi-variable, 1-feature plot, all the contexts are the same as the first
* but keep cx_variable, so that the labels will be correct

	   DO ivar = 2, num_uvars_in_cmnd
	      icxvar = cx_variable(is_cx(ivar))
	      CALL TRANSFER_CONTEXT( is_cx(1), is_cx(ivar) )
	      cx_variable(is_cx(ivar)) = icxvar  
	   ENDDO
	ENDIF

	IF (dsg_as_time) DEALLOCATE (station_index) 
	IF (dsg_as_time) DEALLOCATE (dep_dat) 
	
* successful completion
	RETURN

* error exit2
 5000	RETURN
 5100	CALL ERRMSG(ferr_invalid_command, status,
     .           "For multi-variable PLOT of discrete DSG data, select just one feature", *5000)
 5200   CALL ERRMSG(ferr_invalid_command, status,
     .          "PLOT discrete data, unrecognized Feature type direction", *5000)
 5300	CALL ERRMSG(ferr_invalid_command, status,
     .		"Plot discrete DSG data: variable is not a list of observations", *5000)
 5400	CALL ERRMSG(ferr_invalid_command, status,
     .		"Plot discrete DSG data: too many lines", *5000)
 5500	CALL ERRMSG( ferr_grid_definition, status,
     .		'Data grid is not a DSG grid ',
     .		*5000 )

	END
