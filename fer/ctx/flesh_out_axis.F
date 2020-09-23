	SUBROUTINE FLESH_OUT_AXIS( idim, cx, status )

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
* given a skeletal context with a known grid transfor the subscript coord
* data to world coordinates and world to subscript on the specified axis
* make sure all time coordinates are in time step values (vs. secs from BC)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/26/89 - extracted from FLESH_OUT_CONTEXT
*                - handles axes of uvars where all region info is contained
*		 - detects and set up interpolations
*		 - inserts unspecified limits for file vars
* V230:  7/20/92 - check TM_DATE_OK for valid time axis before doing time xlate
*                - kludge fix to err230_fixed_limits.jnl
*        8/10/92 - point location with transform (e.g. Z=5@DIN) needs
*                  exact location preserved for proper labelling
* V320:	 11/2/94 - checking if interpolation is required needs to be based on
*		   single precision coordinate comparison
* V420:   1/96   - if the context being completed involves modulo
*		   regridding then the interpretation of the limits is altered
* *kob* 11/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V450:   1/97   - fix to never-observed bug: the check
*		        "uvar_given(idim,var) .LE. uvlim_gvn_xact"
*		   ignores the possibility of uvlim_irrelevant
*  *KOB* 9/97    - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in 
*                  order to use cx_calendar to determine style of taxis
* V500 *kob* 3/99- up VAR_CODE decl from 8 to 64 chars
*      *sh*  5/99 - go ahead and flesh out implicit axes (to improve cache hits
*		    when netCDF strides are invoked)
* V510: *sh* 12/99 - support for uvlim_hiddef (EF caching)
*	*sh* 4/00 - since netCDF dsets now produce dynamic axes we cannot
*		    use line .LT. max_lines as a test of an implicit axes
* v5.3 *acm* 1/01 - line_cal_name is part of the data set information
* V530: *sh* 4/01 - added comment about in_get_grid_mode
* V533: *sh* 6/01 - added logic for strides (cx_delta)
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars
* V560:*acm* 4/04 - stride/modulo fixes: use NINT to convert REAL*8 to integer
* *acm* v580 11/04- Keep cx_cal_id, the calendar ID used when computing 
*                  world coordinates
* V580 *acm* 11/04- Fix divide by zero; cx_delta(idim,cx) is 0 if user
*                   incorrectly specifies LIST/I=1:100:0 VAR
* V68  *acm* 1/12  explicit single-precision calculations for consistent results in
*		   for double-precision ferret
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP.
* V683  *acm*10/12  Fix bug 1972: cx_calendar and cx_cal_id are changed only
*                   if it's the T dimension not the T or F dimension.
* V690  *sh* 4/14 For curvilinear axes keep ww lims as-is (may exceed ax lims)
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V720  *sh*  4/17 support DSG grids
* V73+ *acm* 12/17 INIT_CXDSG needs nfeatures
* V741  *acm* 6/18 Fix issue 1868: Let the transform remain in the context when 
*                  variable is a constant. So let a = 2; list a[x=@trans] will no 
*                  longer return 2, but will be either computed (for NGD and NBD),
*                  or will be caught as an error instead of returning 2.
* V741  *acm* 7/18 Issue 1868, do not wipe out the trans setting in the case of a constant
* V74+ *acm*10/10 Partial fix to working with E contexts in dsg datasets.
* V744 *acm* 11/18 Issue 965, change behavior when axis is given with strides in the region
* V761 *acm* 7/20 Issue 1981:in ISUBSCR_CX distinguish rounding for single 
*                 point vs range of points

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'implicit.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'
	include 'xtext_info.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data

* calling argument declarations:
	INTEGER	idim, cx, status

* internal variable declarations:
	LOGICAL TM_DATE_OK, TM_FPEQ, INHERITED_LIMITS, TM_ITSA_DSG_RAGGED,
     .		ITSA_REGRID_FROM_DSG, 
     .		point, lim_redundant, is_time, fromdsg, abst_nolim, 
     .		its_dsg, its_cmpnd
	INTEGER	TM_LENSTR1, ISUBSCR_CX, DSG_WHATS_IT, TM_DSG_DSET_FROM_GRID,
     .		TM_DSG_NFEATURES,
     .		ss, lo_ss, hi_ss, axis, grid, cat, var, ispread,
     .		orientation, obs_dimlen, obs_line, dsgness, taxis,
     .		xlate_grid, dset, e_line, nfeatures, i
	INTEGER TM_GET_CALENDAR_ID, cal_id
	REAL*8	TM_WORLD, SECS_TO_TSTEP, ww_nrst, ww_given, min_ww, max_ww
	CHARACTER	VAR_CODE*128, string*128

* initialize
	var  = cx_variable( cx )
	cat  = cx_category( cx )
	grid = cx_grid    ( cx )
	axis = grid_line( idim, grid )
	status = ferr_ok
	dsgness = DSG_WHATS_IT(grid)
	fromdsg = ITSA_REGRID_FROM_DSG( cx)

* DSG grids (4/17)
* For a DSG contiguous ragged array grid (a DSG observation var) the
* subscript limits are always set to the 1D full limits of the obs axis
* For a DSG instance variable the 1D subscript limits are always set to
* the full span of the E (feature) axis.
* In both observation vars and instance vars if WW limits are provided,
* they are carried along passively to be used at the time of action
* commands (LIST, PLOT, etc.) as a filter on which instances
* of the DSG are to be processed.
* The F axis of a DSG grid, alone, is processed as a typical gridded axis
	IF (dsgness .NE. pdsg_not_dsg .OR. fromdsg) THEN
	   IF (idim .EQ. f_dim) GOTO 1000  ! treat like a gridded axis
* ... initialize DSG info
	   its_dsg = .TRUE.
	   CALL TM_DSG_FACTS(grid, orientation, obs_dimlen, obs_line,
     .                       its_dsg, its_cmpnd)
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   IF (.NOT. its_dsg) dsgness = pdsg_not_dsg
	ENDIF
	IF (dsgness .NE. pdsg_not_dsg ) THEN
* as a friendliness feature make E subscript limits behave as if they
*     were given as WW limits. E=instance=feature axis of a DSG
	   IF ( idim .EQ. e_dim .AND. cx_given(e_dim,cx) ) THEN
	      IF (cx_by_ss(e_dim,cx)) THEN
	         cx_lo_ww(idim,cx) = cx_lo_ss(cx,idim)
	         cx_hi_ww(idim,cx) = cx_hi_ss(cx,idim)
	         cx_by_ss(e_dim,cx) = .FALSE.
	      ENDIF
	   ELSE IF ( idim .EQ. e_dim .AND. cx_lo_ss(cx,idim).NE.unspecified_int4 ) THEN
	      IF (cx_by_ss(e_dim,cx)) THEN
	         cx_lo_ww(idim,cx) = cx_lo_ss(cx,idim)
	         cx_hi_ww(idim,cx) = cx_hi_ss(cx,idim)
	         cx_by_ss(e_dim,cx) = .FALSE.
	      ENDIF
	   ENDIF

* ignore other limits set by subscript (alt: could flag as a user error)
	   cx_lo_ss( cx, idim ) = unspecified_int4 
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_by_ss(idim,cx) = .FALSE.
* if a T axis, convert date-times into grid-specific time step values
* from their representation as negative secs from BC
	   IF ( idim.EQ. t_dim
     .	 .AND. cx_hi_ww(idim,cx) .NE. unspecified_val8
     .	 .AND. cx_calendar(idim,cx)      ) THEN
	     xlate_grid = dsg_xlate_grid(dset)
	     taxis = grid_line(t_dim, xlate_grid)
             cal_id = TM_GET_CALENDAR_ID (line_cal_name(taxis))
             IF ( TM_DATE_OK( line_t0(taxis), cal_id ) ) THEN
	         cx_lo_ww(idim,cx) = SECS_TO_TSTEP
     .                ( xlate_grid, idim, -cx_lo_ww(idim,cx) )
	         cx_hi_ww(idim,cx) = SECS_TO_TSTEP
     .                ( xlate_grid, idim, -cx_hi_ww(idim,cx) )
		 cx_calendar(idim,cx) = .FALSE.
                 cx_cal_id(cx) = cal_id
              ENDIF
	   ENDIF
* constrain the world limits to the coordinate range of the dataset
	   CALL DSG_COORD_LIMS(dset, idim, min_ww, max_ww)
	   IF (min_ww .NE. unspecified_val8 ) THEN
	      IF  (cx_lo_ww(idim,cx) .EQ. unspecified_val8) THEN
	         cx_lo_ww(idim,cx) = min_ww
	         cx_hi_ww(idim,cx) = max_ww
	      ELSE
	         IF (cx_lo_ww(idim,cx) .LT. min_ww) cx_lo_ww(idim,cx) = min_ww 
	         IF (cx_hi_ww(idim,cx) .GT. max_ww) cx_hi_ww(idim,cx) = max_ww 
	      ENDIF
	   ENDIF
* set the subscript limits to capture the full 1D netCDF variable
	   IF (dsgness .EQ. pdsg_instance) THEN
	      IF (idim .EQ. e_dim
     .	    .AND. cx_trans(idim,cx) .LE. trans_compress_code) THEN
	         cx_lo_ss(cx,idim) = 1
	         cx_hi_ss(cx,idim) = line_dim(axis) ! E axis length
	      ENDIF
	   ELSEIF (dsgness .EQ. pdsg_observation) THEN
	      IF (its_cmpnd) orientation = z_dim
	      IF (idim .EQ. orientation) THEN
	         cx_lo_ss(cx,idim) = 1
	         cx_hi_ss(cx,idim) = obs_dimlen   ! DSG observation dim length
	      ENDIF
	   ENDIF
* constraining E axis implicitly constrains the XYZT dimensions
* For example E=5 of a trajectory collection has
* only the XYZT extent of that trajectory -- not of the full collection
	   IF ( idim .EQ. e_dim .AND. cx_given(e_dim,cx) ) THEN
	      nfeatures = TM_DSG_NFEATURES( grid )
	      CALL INIT_CXDSG(dset, cx, nfeatures)
	      CALL APPLY_DSG_E_CONSTRAINTS(dset, cx)
	   ENDIF
! DSGTBD: constraining the X axis of a DSG implicitly constrains YZT, too
! because it may remove features that possess the extreme values in YZ or T
! Above we have addressed only explicit constraints on E.
! So as an optimization one might consider recomputing the limits any time
! any of the constraints are altered.
	   GOTO 4000
* When regridding *from* a DSG grid, the ww limits of the DSG become irrelevant
* Since the DSG context will be available at the time of regridding, the
* information contained in these limits will be available
	ELSEIF (fromdsg) THEN
* Regridding *from* a DSG
* Leave the E and orientation axes as-is so that user-specified limts will
* by retained  -- e.g. LIST/L=1:10 vdsg[GT=new_tax]
	   IF (idim.EQ.e_dim .OR. idim.EQ.orientation) THEN
	      CONTINUE
!	      cx_lo_ss(cx,idim) = 1
!	      cx_hi_ss(cx,idim) = line_dim(axis)
	   ELSE
	      cx_by_ss(idim,cx) = .TRUE.
	      cx_lo_ss(cx,idim) = unspecified_int4
	      cx_hi_ss(cx,idim) = unspecified_int4
	      cx_lo_ww(idim,cx) = unspecified_val8  ! unnecessary?
	      cx_hi_ww(idim,cx) = unspecified_val8  ! unnecessary?
	   ENDIF
	ENDIF
! DSGTBD need to test a variety of target grids - suitable and unsuitable
* *** DSG grid end


* For setting the calendar information
 1000	   is_time = idim.EQ.t_dim .OR. idim.EQ.f_dim

* check for special actions needed on user-defined variables
* ( e.g. how to handle U[K=5] - U[K=1:10] )
	IF ( cat .EQ. cat_user_var ) THEN
	   lim_redundant = uvar_given( idim, var ) .LE. uvlim_gvn_xact
     .		     .OR.  uvar_given( idim, var ) .EQ. uvlim_irrelevant
     .		     .OR.  uvar_given( idim, var ) .EQ. uvlim_hidden
	ELSE
	   lim_redundant = .FALSE.
	ENDIF

* for modulo regridding the apparent limits supplied in the definition
* are "swallowed" up by the regridding process. Externally applied limits
* **should** be applied (e.g. A[GT=climatology@MOD,L=1:3] where A=SST[L=lo:hi])
	IF ( .NOT.INHERITED_LIMITS(idim,cx) )  lim_redundant = .FALSE.

	abst_nolim = .FALSE.
	IF ( line_name(axis) .EQ. 'ABSTRACT' .AND. var.LE.max_uvar) THEN
	   abst_nolim = ( cx_trans(idim,cx) .NE. trans_no_transform
     .    .AND. uvar_given(idim,var) .EQ. uvlim_irrelevant ) 
             IF ( cx_trans(idim,cx) .NE.  trans_variance .AND.
     .            cx_trans(idim,cx) .NE.  trans_shift .AND.
     .            cx_trans(idim,cx) .NE.  trans_deriv_cntr .AND.
     .            cx_trans(idim,cx) .NE.  trans_deriv_fwrd .AND.
     .            cx_trans(idim,cx) .NE.  trans_deriv_bkwd .AND.
     .            cx_trans(idim,cx) .NE.  trans_integ_def .AND.
     .            cx_trans(idim,cx) .NE.  trans_integ_indef .AND.
     .            cx_trans(idim,cx) .NE.  trans_locate .AND.
     .            cx_trans(idim,cx) .NE.  trans_wgted_eq .AND.
     .            cx_trans(idim,cx) .NE.  trans_nrst_dist_abv .AND.
     .            cx_trans(idim,cx) .NE.  trans_nrst_dist_blw .AND.
     .            cx_trans(idim,cx) .NE.  trans_nrst_indx_abv .AND.
     .            cx_trans(idim,cx) .NE.  trans_nrst_indx_blw .AND. 
     .            cx_trans(idim,cx) .NE.  trans_event_mask .AND.
     .            cx_trans(idim,cx) .NE.  trans_std_dev .AND.
     .            cx_trans(idim,cx) .NE.  trans_avg_wgt) abst_nolim = .FALSE.

	ENDIF

	IF ( axis .EQ. mnormal ) THEN
* THIS AXIS IS NORMAL TO THE DEFINING GRID
* wipe clean any stray information about it
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx ) = unspecified_val8
	   cx_hi_ww( idim, cx ) = unspecified_val8
	   cx_delta( idim, cx ) = unspecified_val8
	   cx_trans( idim, cx ) = trans_no_transform
* kob 9/97 - need to reset cx_calendar as well

	   IF (is_time) cx_calendar(idim,cx) = .FALSE.

* axes with fully "given" regions will have their regions determined via
* MERGE_CONTEXT (when computed) or PASS_AMBIG_LIMS (when found in memory)
* 5/99 - added ".AND. axis .LT. max_lines" -- we will go ahead with
* fleshing out implicit axes (e.g. a[i=10:100:10])
* to improve the cache hit rate when netCDF strides are used. Note the peril
* in this is the behavior of a[i=3:5] where a=b[i=10:100:10]
* 4/00: use line_class instead of max_lines as the test
	ELSEIF ( .NOT.cx_given(idim,cx) .AND. lim_redundant )  THEN
*
* 11/2018 acm: for issue 965, remove this part of the test 
* .		.AND. line_class(axis) .NE. pline_class_stride ) THEN

!pre 4/00     .		.AND. axis .LT. max_lines ) THEN

****
* 7/92 kludge:
* NOTE (4/01): logical variable in_get_grid_mode would be a more reliable soln
* This routine is called by is_uvar_grid (while determining the grid)
* BEFORE uvar_given has been determined.  When called from there lim_redundant
* may be invalid.  Since uvar_given is initialized to uvlim_irrelevant we will
* use this as value a test of whether the call is from is_uvar_grid
* Probably it would be better to make this check while determining
* lim_redundant, above, but that would be a more significant alteration of
* working code - so this is the chicken-out solution
* WAS:	   IF ( cx_trans(idim,cx) .NE. trans_no_transform ) GOTO 5000

* 1/97 - might this kludge have been rendered unnecessary by the 1/97 change to
*        the determination of lim_redundant??
****
          IF ( cx_trans(idim,cx)    .NE. trans_no_transform
     .    .AND. uvar_given(idim,var) .NE. uvlim_irrelevant ) GOTO 5000

****
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx ) = unspecified_val8
	   cx_hi_ww( idim, cx ) = unspecified_val8
	   cx_delta( idim, cx ) = unspecified_val8
* kob 9/97 - need to reset cx_calendar as well
	   IF (is_time) THEN
              cx_calendar(idim,cx) = .FALSE.
              cx_cal_id(cx) = unspecified_int4
           ENDIF

	ELSEIF ( cx_by_ss( idim, cx )) THEN!   .AND. .NOT.abst_nolim
* POSITIONS WERE SPECIFIED BY SUBSCRIPT
* is there valid position data for this axis ?
	   IF ( cx_lo_ss( cx, idim ) .EQ. unspecified_int4 ) THEN
* ... no position data but maybe it's a file variable with only 1 point
	      CALL VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )
! old	      IF ( lo_ss.EQ.unspecified_int4 .OR. lo_ss.NE.hi_ss ) RETURN
	      IF ( lo_ss.EQ.unspecified_int4 ) RETURN
	      cx_lo_ss(cx,idim) = lo_ss
	      cx_hi_ss(cx,idim) = hi_ss
	    ENDIF

* if strides in use then set the upper subscript to correct parity
	    IF (cx_delta(idim,cx) .NE. unspecified_val8) THEN
	      IF (cx_delta(idim,cx) .LE. 0.) GOTO 5300
	      ispread = cx_hi_ss(cx,idim) - cx_lo_ss(cx,idim)
	      ispread = (ispread/NINT(cx_delta(idim,cx)))
     .		      * NINT(cx_delta(idim,cx))
	      cx_hi_ss(cx,idim) = cx_lo_ss(cx,idim) + ispread
	    ENDIF

* compute world positions as grid box edges
	   cx_lo_ww(idim, cx) = TM_WORLD( cx_lo_ss( cx,idim),
     .					  grid, idim, box_lo_lim )
	   cx_hi_ww(idim, cx) = TM_WORLD( cx_hi_ss( cx,idim),
     .					  grid, idim, box_hi_lim )

	   IF ( is_time .AND. cx_calendar(idim,cx) ) THEN
             cal_id = TM_GET_CALENDAR_ID (line_cal_name(axis))
             cx_cal_id(cx) = cal_id
           ENDIF


	ELSEIF ( cx_aux_stat( idim, cx ) .EQ. paux_stat_false_1pt_regrid
     .	   .AND. cx_unstand_grid( cx		)	             ) THEN
* 4/14 - WORLD COORDINATES ON A CURVILINEAR OR LAYERZ AXIS
	      cx_lo_ss(cx,idim) = unspecified_int4
	      cx_hi_ss(cx,idim) = unspecified_int4
*  leave ww limits as-is. Values refer to the aux var range, not the native axis 
	        CONTINUE

	ELSE
* WORLD COORDINATE POSITIONS
* is there valid position data for this axis ?
	   IF ( cx_lo_ww( idim, cx ) .EQ. unspecified_val8 ) THEN
* ... no position data but maybe it's a file variable with only 1 point
	      CALL VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )
! old	      IF ( lo_ss.EQ.unspecified_int4 .OR. lo_ss.NE.hi_ss ) RETURN
	      IF ( lo_ss.EQ.unspecified_int4 ) RETURN
	      cx_lo_ss(cx,idim) = lo_ss
	      cx_hi_ss(cx,idim) = hi_ss
* ... compute world positions as grid box edges
	      cx_lo_ww(idim, cx) = TM_WORLD( lo_ss, grid, idim, box_lo_lim )
	      cx_hi_ww(idim, cx) = TM_WORLD( hi_ss, grid, idim, box_hi_lim )
	   ENDIF

	   point = cx_lo_ww(idim,cx) .EQ. cx_hi_ww(idim,cx)

* convert times given as seconds from BC to time step values
* *kob* 8/97 - use cx_calender instead of cx_lo_ww

	   IF ( is_time .AND. cx_calendar(idim,cx) ) THEN
             cal_id = TM_GET_CALENDAR_ID (line_cal_name(axis))
             IF ( TM_DATE_OK( line_t0(axis), cal_id ) ) THEN
	         cx_lo_ww(idim,cx) =
     .                SECS_TO_TSTEP( grid, idim, -cx_lo_ww(idim,cx) )
	         cx_hi_ww(idim,cx) =
     .                SECS_TO_TSTEP( grid, idim, -cx_hi_ww(idim,cx) )
		 cx_calendar(idim,cx) = .FALSE.
                 cx_cal_id(cx) = cal_id
              ENDIF
	   ENDIF
	      
* world coordinate is a single point
	   IF ( point ) THEN
	      ww_given = cx_lo_ww(idim,cx)
	      ss = ISUBSCR_CX(ww_given,grid,idim,cx,round_sngl)
	      ww_nrst = TM_WORLD( ss, grid, idim, box_middle )
* ... need to interpolate ? 
	      IF (  mode_interpolate
     .	     .AND. .NOT.TM_FPEQ((ww_given),(ww_nrst)) ) THEN
* ... set up context to interpolate
	         IF ( cx_trans(idim,cx) .NE. trans_no_transform
     .		.AND. cx_trans(idim,cx) .NE. trans_interpolate ) GOTO 5200
	         cx_trans(idim,cx) = trans_interpolate
	      ELSEIF ( cx_trans(idim,cx) .EQ. trans_no_transform ) THEN
* ... interpolation not desired, needed or possible
	         cx_lo_ww(idim,cx) = ww_nrst
	         cx_hi_ww(idim,cx) = ww_nrst
	         cx_lo_ss(cx,idim) = ss
	         cx_hi_ss(cx,idim) = ss
	      ELSE 
* ... keep original (exact) limits if transformed axis ! 8/92
	         cx_lo_ss(cx,idim) = ss
	         cx_hi_ss(cx,idim) = ss
	      ENDIF
	   ELSE

* world coordinates span a range
* ... compute subscripts ( rounding inward to resolve ambiguities )
	      cx_lo_ss( cx, idim ) = ISUBSCR_CX
     .				(cx_lo_ww(idim, cx), grid, idim, cx,round_up)
	      cx_hi_ss( cx, idim ) = ISUBSCR_CX 
     .				(cx_hi_ww(idim, cx), grid, idim, cx,round_dn)
	   ENDIF
	ENDIF

* TRANSFORMATION REDUCES RANGE TO A POINT ?
	IF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_by_ss( idim, cx ) = .FALSE.
	ENDIF

 4000   CONTINUE

* successful completion
	RETURN

* error exits
 5000	RETURN

 5200	string = VAR_CODE( cx_category(cx), cx_variable(cx) )
	CALL ERRMSG( ferr_trans_nest, status,
     .		'cannot perform interpolation '//pCR//
     .		ww_dim_name(idim)//' axis of '//string(:TM_LENSTR1(string))//
     .		' is already transformed'//pCR//
     .		'Can you define an intermediate variable ?',
     .		*5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'delta non-positive on '//ww_dim_name(idim)
     .		//' axis of '//VAR_CODE(cx_category(cx),cx_variable(cx)),
     .						*5000)

	END
