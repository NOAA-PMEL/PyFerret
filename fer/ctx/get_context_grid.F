	SUBROUTINE GET_CONTEXT_GRID ( cx, status )

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
* determine the grid to use for a single variable evaluation
* the given context must have the logical cx_unstand_grid set and
* if .TRUE. it must have a cx_grid value

* (9/95) if cx_has_impl_grid is set then we must reconcile the grid that
* is implied (cx_impl_grid) with a grid that may or may not have been
* explicitly mentioned (cx_grid).
*	If they match -- return the explicit grid
*	If they don't -- complete the implicit grid definition
* If an implicit grid matches an existing non-implicit grid return that,
* else "allocate" an implicit grid (either by creating one or by incrementing
* the usage count for that grid)
* Also, if delta specifications were used (e.g. var[i=1:100:10]) then an
* implicit axis AND grid must be created.


* FERRET DOCUMENTATION: MANAGING DYNAMIC GRIDS AND AXES:  OVERVIEW

*************************************
* *** MANAGING DYNAMIC GRIDS AND AXES:  OVERVIEW  ***

* see also explanations of TMAP library grid and axis allocation schemes
* in routines TM_ALLO_DYN_GRID and TM_ALLO_DYN_LINE

* Implicit (dynamic) grids and axes must be tracked so that 1) they
* are deleted when no longer needed and 2) they are never deleted
* while still in use.  Deletion makes the "slot" available for re-use.
* The information indicating when a dynamic grid or line is needed is
* contained in its "use count" (use_cnt).  At various points the terms
* "incrementing", "allocating" and "using" may be used synonymously.
* These refer to TM_ALLO_DYN_... and TM_USE_.... Similarly,
* "decrementing" is the same as TM_DEALLOC_DYN_...  Deallocation may
* or may not be the same as "deletion" of a grid or line -- it will be
* deleted only if its usage count falls to zero.

* Grids have their use count incremented for each user variable (or
* intermediate result) that depends on the grid.  The "intermediate
* result" refers to entities that come into existence only briefly
* during the evaluation of an expression.  Take, for example, the
* expression "VAR[gx=var2] + 5":  during the "get_grid" phase a
* dynamic grid is determined for this expression and pointed to by
* uvar_grid.  As a result of this "usage" the grids use-cnt is
* incremented (from 0 to 1 if it is newly created).  Now during the
* evaluation of the expression the intermediate result "VAR[gx=var2]"
* will be passed to to GET_VAR_CONTEXT.  The routines GET_CONTEXT_MODS
* and GET_CONTEXT_GRID will determine a grid for this sub-expression.
* Since this grid happens to be the same as the grid of the full
* expression (which differs only in the addition "+5") the use count
* will be momentarily incremented (from 1 to 2) while this
* intermediate result exists on the stack.

* Usage of dynamic grids by intermediate results is tracked with the
* boolean cx_has_impl_grid -- i.e. responsibility for incrementing and
* decrementing lies with the **context**.  This responsibility may be
* passed from one intermediate result to another.

* A lines (axis) has its use count incremented whenever a grid uses
* the line or whenever another (dynamic) line is created as a child of
* this line.  Correspondingly the use count is decremented (resulting
* in deletion if it falls to zero) whenever the corresponding grid or
* child axis is deleted.  The responsibility in the grids is indicated
* by grid_line(idim,grid) > max_lines.  The responsibility in child
* lines is indicated in line_parent.  The responsibility for the line
* may be transferred from one grid to another (see calls to
* TM_COPY_GRID_W_LINE_USE)
* Note: as of 4/00 ALL grids and axes maintain use counts

*************************************


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 10/14/86
* revision 0.1 - 2/18/88 - checks for unspecified grid ( used to flag computed
*			   variables that are non-GFDL/205 output )
* revision 0.2 -  4/7/88 - improved error message for calculated vars
* revision 0.3 -  4/1/89 - determine regridding transformations

* V230: 8/23/92 - impose xyave regrid if appropriate (was in GET_CONTEXT_MODS)
* V420: 9/95	- Support for implicit grid: flesh out the grid here
*	10/95	- support for implicit axes: create axes here and insert
*		  into implicit grid
*	 1/96	- idim subscripting bug fixed 1/96 ... how did it escape?
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
* V500 *kob* 3/99- up VAR_CODE decl from 8 to 64 chars
* V510: *sh* 9/99 - bug revealed while implementing SAMPLE* functions
*		if cx_unstand_grid is FALSE on coming into this routine
*		then the basic grid for the result shoud be std_grid. The
*		value of use_grid upon entry cannot be trusted to be correct
*	*sh* 4/00 - since netCDF dsets use dynamic grids now, all grids and
*		  lines maintain a use count
* V520: 6/00 *sh* - bug fix - if the quick exit is taken here because
*		std_grid is not yet known the axes within the implicit grid
*		were never allocated so dont deallocate them
* V532: *sh* 6/01 - bug fix: err522_impl_ax_ignored.jnl -- the variable
*		impl_grid wasn't initialized if cx_impl_grid was set
*		in get_context_mods by gx=name
* V533: *sh* 6/01 - set regrid transform for strides to prgrd_xact_pts
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP.
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V690 12/13 *sh* set default regrid transforms for aux var regridding transform
*	          set up dynamic axis if cx_delta=pauto_1pt_ax
*      4/14  *sh* return standard grid if "false regridding" - 1 pt curvi
* V695  *acm* 2/15 cx_calendar stores info about formatting of T and F axes
* V695+  *sh* 3/15 assign regrid transform for T-F plane FMRC regridding
* V698+  *sh* 5/15 ax units for gz(density)=lo:hi:del should come from aux var
*		   only partially fixed here: T0 not parsed from auxvar units
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V698  3/16 *sh* added logic for cat_ef_time2d (ACTS_LIKE_FVAR)
*                 more sophisticated logic to determine an implied "@FMRC"
* V720  5/17 *sh* added support for DSG grids 
* V74+  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
* V76  1/20 *acm* working with Point-type dsg data


        IMPLICIT NONE
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'interp_stack.parm'
	include 'tmap_dims.parm'
#   include "tmap_dset.parm"
	include 'xcontext.cmn'
	include 'xalgebra.cmn'
	include 'xtext_info.cmn'
	include 'xprog_state.cmn'
	include 'implicit.parm'
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xfr_grid.cmn'   ! stack ptr should be in TMAP lib
	include 'xunits.cmn_text'

* calling argument declarations:
	INTEGER	cx, status

* internal variable declarations:
        LOGICAL   GEOG_COS_FACTOR, GEOG_LABEL, ACTS_LIKE_FVAR,
     .		  ITS_FMRC_2DTIME, TM_ITSA_DSG, REGRID_TO_SELF,
     .		  its_dyn, has_impl_axis, explicit_unstd, use_std_grid,
     .		  made_aux_line_dummy, at_fmrc, dsg_pts
	INTEGER	  KNOWN_GRID, TM_UNIT_ID,
     .		  DSG_WHATS_IT, TM_DSG_DSET_FROM_GRID, TM_DSG_INSTANCE_GRID,
     .		  TM_DSG_OBS_GRID,
     .		  use_grid, ilo, ihi,
     .		  cat, var, dset, std_grid, std_axis, use_axis, idim,
     .		  base_grid, impl_grid, src_line, i,
     .		  dsgness, orientation, rgrd_xform
	REAL*8	  SECS_TO_TSTEP, dlo, dhi, delta
	CHARACTER VAR_CODE*128, AUX_VAR_UNITS*64

* initialize
	made_aux_line_dummy = .FALSE.
	cat  = cx_category( cx )
	var  = cx_variable( cx )
	dset = cx_data_set( cx )
        use_grid = cx_grid( cx )
	explicit_unstd = cx_unstand_grid( cx ) ! 9/99

* by this point the variable and data set must be known
	IF ( var .EQ. unspecified_int4 )	CALL ERRMSG
     .			( ferr_syntax,status,'no variable specified',*5000 )
	IF (  ACTS_LIKE_FVAR(cat) .AND. dset.EQ.unspecified_int4 )
     .	  CALL ERRMSG( ferr_state_not_set,status,'SET DATA_SET',*5000 )

* determine the standard grid for this variable
	std_grid = KNOWN_GRID( dset, cat, var )

* if the standard grid isn't known and an implicit grid has been requested
* then return to the stack to get it
	IF ( std_grid .EQ. unspecified_int4 ) THEN
	   IF ( cx_has_impl_grid(cx) ) THEN
#ifdef debug_dyn_grids
	      IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('dealloc ',isact_class_impl_grid,cx_impl_grid(cx),0)
#endif
* ... grid was barely started by GET_CONTEXT_MODS. Axes were never allocated
	      CALL DEALLO_UNFORMED_GRID(status)
	   ENDIF
	   RETURN
	ENDIF

* **** DSG grid handling ****
* The grid of a DSG variable is always known / never implicit
! DSGTBD: Case of regridding *from* a DSG e.g. "t_25[T=@AVE,GT=uniform_Tax]" 
	IF (std_grid .NE. unspecified_int4) THEN
	   IF (.NOT.explicit_unstd .AND. TM_ITSA_DSG(std_grid) ) THEN
* Handle compressing transformation applied to a DSG grid
	      dsgness = DSG_WHATS_IT(std_grid)
	      dset = TM_DSG_DSET_FROM_GRID(std_grid)! always same as cx (?)
	      orientation = dsg_orientation(dset)
	      IF (orientation.EQ.pfeatureType_TimeseriesProfile .OR. 
     .		  orientation.EQ.pfeatureType_TrajectoryProfile) orientation = z_dim
* ... trap transformations that are nonsense on ragged arrays
*     transforming "across" (rather than along) the obs axis is nonsense
	      DO idim = 1, e_dim
	         IF (cx_trans(idim,cx) .NE. trans_no_transform) THEN
	            IF (dsgness .EQ. pdsg_observation) THEN
	               IF (idim .NE. orientation) GOTO 5410
	            ELSEIF (dsgness .EQ. pdsg_instance) THEN
	               IF (idim .NE. e_dim      ) GOTO 5420
	            ENDIF
	         ENDIF
	      ENDDO
	      IF (dsgness .EQ. pdsg_observation) THEN
	         IF ( cx_trans(orientation,cx) .GT. trans_compress_code ) THEN
* ... If a compressing transformation is applied along the "orientation" axis
*     (e.g. T=@AVE on a Tseries feature collection)
*     the result is a single point per feature -- i.e. an instance variable.
*     Handle this by replacing the obs grid with the instance grid.
*     This allows FLESH_OUT_AXIS, DSG_WHATS_IT, etc. to perform correctly on a
*     orientation-compressed result with no special code.
	            std_grid = TM_DSG_INSTANCE_GRID(dset)
	         ENDIF	
	      ENDIF
	   ENDIF
	ENDIF
	IF (use_grid .NE. unspecified_int4) THEN
	   IF (explicit_unstd .AND.TM_ITSA_DSG(use_grid) ) THEN
* Handle regridding *to* a DSG grid
	      dsgness = DSG_WHATS_IT(use_grid)
	      dset = TM_DSG_DSET_FROM_GRID(use_grid)
	      orientation = dsg_orientation(dset)
* ... error check: source data must exist along the orientation axis
	      IF (orientation.NE.pfeatureType_Point .AND. 
     .		  grid_line(orientation,std_grid) .EQ. mnormal) GOTO 5430
* ... as a friendliness freature substitute the obs grid if they gave
*     an instance variable as the target
	      IF (grid_line(orientation,use_grid) .EQ. mnormal) THEN
	         use_grid = TM_DSG_OBS_GRID(dset)
	         cx_grid( cx ) = use_grid
	      ENDIF
* ... set the regridding transform
! DSGTBD: multiple regridding algorithms not yet supported
!     currently chooses the source grid value of the box holding the target pt
!     Reused "prgrd_nearest". For unequal spaced grid is it always "nearest"?
!     Linear interpolation option is definitley needed and should be default.
	      rgrd_xform = prgrd_nearest
* ... flag the axes to be involved in the regridding
! DSGTBD: user control over which axes are regridded: eg "GXYT=target"
!     should allow the Z axis to be excluded: "mode_temp[Z=0,GXYT=my_DSG]"
!     May require hacks in GET_CONTEXT_MODS.
	      DO idim = 1, t_dim
	         IF (dsg_coord_var(idim,dset) .EQ. int4_init) THEN
	            cx_regrid_trans(idim, cx) = unspecified_int4
	         ELSE
	            cx_regrid_trans(idim, cx) = rgrd_xform
	         ENDIF
	      ENDDO
	      cx_regrid_trans(e_dim, cx) = unspecified_int4
	      RETURN
	   ENDIF
	ENDIF
* SEE REGRIDDING **FROM** DSG GRID AT END

* Are we gridding from a points-type dsg dataset?
	dsg_pts = .FALSE.
	IF (TM_ITSA_DSG(std_grid)) THEN
	    orientation = dsg_orientation(dset)
	    dsg_pts = dsg_orientation(dset).EQ.e_dim .OR. dsg_orientation(dset).EQ.0
	ENDIF
* **** END OF DSG handling ****

* regridding implied through delta values or special transformations?
	has_impl_axis = .FALSE.
	DO 10 idim = 1, nferdims
	   IF (cx_delta(idim,cx) .NE. unspecified_val8) THEN
* ... make sure the specified delta makes sense (this test moved 1/96)
	      IF ( cx_delta(idim,cx) .NE. unspecified_val8
     .	     .AND. cx_delta(idim,cx) .NE. pauto_1pt_ax
     .	     .AND. cx_delta(idim,cx) .LE. 0.0D0		) GOTO 5200
	      has_impl_axis = .TRUE.
	   ENDIF
 10	CONTINUE
	IF ( has_impl_axis ) THEN

* ... we need an implicit grid to hold the axes so make it if we don't have one
	   IF ( .NOT.cx_has_impl_grid(cx) ) THEN
	      CALL ALLO_GRID_TO_CX( cx, impl_grid, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	      cx_unstand_grid(cx) = .TRUE.
	   ELSE
	      impl_grid = cx_impl_grid(cx)  ! allo'd in get_context_mods 
	   ENDIF
* ... make the implicit axes and save them in the implicit grid
	   DO 20 idim = 1, nferdims
	      IF (cx_delta(idim,cx) .NE. unspecified_val8) THEN
! ... trap for twice-defined axis of implicit grid is removed because
!	var[g=var2,gx=lo:hi:del] needs to be a legal expression
!	         IF ( grid_line(idim,impl_grid).NE.unspecified_int4 )
!     .							GOTO 5100
	         IF ( cx_by_ss(idim,cx) ) THEN
* ... clip to axis ss limits
	            CALL GRID_SUBSCRIPT_EXTREMES(ilo, ihi, std_grid, idim)
		    IF (cx_lo_ss(cx,idim) .LT. ilo ) THEN	! ilo is 1
! unused code: "modulo clip" -> ss #1 to be a multiple of delta from users strt
!	               n = 1 - INT(cx_lo_ss(cx,idim)/cx_delta(idim,cx))
!	               dlo = cx_lo_ss(cx,idim) + n * cx_delta(idim,cx)
	               dlo = 1
	            ELSE
	               dlo = cx_lo_ss(cx,idim)
	            ENDIF
	            dlo = MAX(ilo, cx_lo_ss(cx,idim))
	            dhi = MIN(ihi, cx_hi_ss(cx,idim))
	            IF ( dlo .GT. dhi ) GOTO 5300
	            CALL TM_GET_LIKE_DYN_LINE(	idim,
     .						dlo,
     .						dhi,
     .						cx_delta(idim,cx),
     .						grid_line(idim,std_grid),
     .						pline_class_stride,
     .						grid_line(idim,impl_grid),
     .						status)
	            cx_lo_ss(cx,idim) = 1
	            cx_hi_ss(cx,idim) = line_dim(grid_line(idim,impl_grid))
	            IF (cx_regrid_trans(idim,cx) .EQ. unspecified_int4 )
     .			 cx_regrid_trans(idim,cx) = prgrd_xact_pts
	         ELSE
* ... dont clip to axis world limits
* Note: It would be nicer not to clip but instead to allow any new size of
*	dynamic axis.  Some time spent on this produced "explicit limit"
*	errors from PASS_AMBIG_LIMS -- but only for irregular axis.  See
*	try_impl_line_{ok,bad}.jnl in $fsrc/errors
* ... copy into buffers because dates need to be modified to time steps
	            dlo = cx_lo_ww(idim,cx)
	            dhi = cx_hi_ww(idim,cx)
	            delta = cx_delta(idim,cx)
	            IF (delta .EQ. pauto_1pt_ax) THEN
	               IF (dlo .EQ. dhi) THEN
* generate a single point axis.  The delta of such a line is arbitrary.
	                  IF (dlo .EQ. 0.0) THEN
	                     delta = 0.00001
	                  ELSE
	                     delta = dlo * 0.00001
	                  ENDIF
	               ELSE
	                  delta = dhi - dlo
	                  dlo   = (dlo + dhi)/2
	                  dhi   = dlo 
	               ENDIF
	            ENDIF
	            IF (idim.EQ.t_dim .OR. idim.EQ.f_dim) THEN
* 8/97 *kob* use cx_calender to check if time is date rather than time step
	               IF ( cx_calendar(idim,cx) ) THEN
	                  IF (.NOT.GEOG_LABEL(idim,std_grid)) GOTO 5400
	                  dlo = SECS_TO_TSTEP(std_grid,idim,-1.*cx_lo_ww(idim,cx))
	                  dhi = SECS_TO_TSTEP(std_grid,idim,-1.*cx_hi_ww(idim,cx))
			  delta = delta * 3600
     .				/ line_tunit(grid_line(idim,std_grid)) !was hrs
	               ENDIF
	            ENDIF
* 5/15 ax units for gz(density)=lo:hi:del come from aux var
	            IF ( cx_naux(cx) .GT.0
     .		   .AND. cx_aux_var(idim,cx) .NE. unspecified_int4 ) THEN
* ... set up specimen line using units information from the auxiliary variable
	               CALL TM_MAKE_BASIC_AXIS('tmp_aux_line_dummy',0.,1.,1,
     .		       src_line, status)
	               made_aux_line_dummy = .TRUE.
	               line_units(src_line) = AUX_VAR_UNITS(cx,idim)
	               line_unit_code(src_line) = TM_UNIT_ID(line_units(src_line))
	               line_tunit(src_line) = un_convert(line_unit_code(src_line))
* ** MORE TO BE DONE:  line_t0 and calendar need to be set as in cd_get_1_axis.F
	            ELSE
	               src_line = grid_line(idim,std_grid)
	            ENDIF
	            CALL TM_GET_LIKE_DYN_LINE(	idim,
     .						dlo,
     .						dhi,
     .						delta,
     .						src_line,
     .						pline_class_basic,
     .						grid_line(idim,impl_grid),
     .						status)
	         ENDIF
	         IF ( status .NE. ferr_ok ) GOTO 5000
!! this commented out block didn't work because an unspecified region
!!   lead to the SET REGION default being imposed with ss limits in conflict
!!   with the dynamic axis dimension
!! Set up to look like regridding was requested without the limits specified
!! This is so there is no conflict of explicit limits in cases like
!!     LET a = var[i=lo1:hi1:del];   LIST a[i=lo2:hi2]
!	         cx_lo_ss(cx,idim) = unspecified_int4
!	         cx_hi_ss(cx,idim) = unspecified_int4
!	         cx_lo_ww(idim,cx) = unspecified_val8
!	         cx_hi_ww(idim,cx) = unspecified_val8
!	         cx_given(idim,cx) = .FALSE.
	         cx_delta(idim,cx) =  unspecified_val8
	      ENDIF
 20	   CONTINUE
	ENDIF

* use the standard grid ?
* 4/14 cx_unstand_grid may indicate a "false regrid" used for single point curvi results
* note that false regridding may not have been checked in all possible combos of
* delta values on other axes, etc.
	use_std_grid = .NOT. cx_unstand_grid( cx )
	DO idim = 1, nferdims
	   IF (cx_aux_stat(idim,cx) .EQ. paux_stat_false_1pt_regrid)
     .			use_std_grid = .TRUE.
	ENDDO
	IF ( use_std_grid ) THEN
	   cx_grid( cx ) = std_grid
	   RETURN
	ENDIF

* handle implicit (dynamic) grid creation
	IF ( cx_has_impl_grid(cx) ) THEN
* ... build a (temporary) implicit grid (9/99 logic mod)
	   IF ( explicit_unstd
     .	  .AND. use_grid .NE. unspecified_int4 ) THEN
	      base_grid = use_grid
	   ELSE
	      base_grid = std_grid
	   ENDIF
	   impl_grid = cx_impl_grid(cx)
* ... any axis that wasn't given by the user comes from the source grid
	   DO 50 idim = 1, nferdims
	      IF ( grid_line(idim,impl_grid).EQ.unspecified_int4 )
     .		   grid_line(idim,impl_grid) = grid_line(idim,base_grid)
 50	   CONTINUE
	   use_grid = impl_grid

* increment the use count of dynamic axes that are in-use by the temporary grid
	   DO 60 idim = 1, nferdims
 60	   CALL TM_USE_LINE(grid_line(idim,impl_grid))
	ENDIF

* 9/95 check for incompatibilities in grid geometry -- requiring implicit grid
* (i.e. chk for a normal axis in one grid without same in other)



* Are we gridding from a points-type dsg dataset?  Make a grid that's 
* source-grid by E-dim of DSG using just the first non-normal axis encountered.
	IF (dsg_pts) THEN
	   DO idim = 1, t_dim
	      IF (grid_line(idim,use_grid).EQ.mnormal) CYCLE

* ... allocate and initialize some work space if we don't already have it
	      IF (.NOT.cx_has_impl_grid(cx)) THEN 
	         CALL ALLO_GRID( impl_grid,status )
	         IF (status .NE. ferr_ok) GOTO 5000
	         CALL TM_COPY_GRID_W_LINE_USE(std_grid, impl_grid)
	         grid_name(impl_grid) = '(conformable)'
	         cx_has_impl_grid(cx) = .TRUE.
!	         cx_impl_grid(cx) = impl_grid
#ifdef debug_dyn_grids
                 IF ( mode_diagnostic )  CALL DIAG_OP
     .              ('allocate', isact_class_impl_grid, impl_grid, 0)
#endif
	      ENDIF
	      grid_line(idim,impl_grid) = grid_line(x_dim,use_grid)
	      CALL TM_USE_LINE(grid_line(idim,impl_grid))
	      EXIT ! from loop
	   ENDDO

	ELSE

	   DO 70 idim = 1, nferdims
	      IF (  (grid_line(idim,std_grid).EQ.mnormal)
     .       .NEQV. (grid_line(idim,use_grid).EQ.mnormal) )   THEN
* ... allocate and initialize some work space if we don't already have it
	         IF (.NOT.cx_has_impl_grid(cx)) THEN 
	            CALL ALLO_GRID( impl_grid,status )
	            IF (status .NE. ferr_ok) GOTO 5000
	            CALL TM_COPY_GRID_W_LINE_USE(use_grid, impl_grid)
	            grid_name(impl_grid) = '(conformable)'
	            cx_has_impl_grid(cx) = .TRUE.
!	            cx_impl_grid(cx) = impl_grid
#ifdef debug_dyn_grids
                    IF ( mode_diagnostic )  CALL DIAG_OP
     .              ('allocate', isact_class_impl_grid, impl_grid, 0)
#endif
	         ENDIF
	         CALL TM_DEALLO_DYN_LINE(grid_line(idim,impl_grid))
	         grid_line(idim,impl_grid) = grid_line(idim,std_grid)
	         CALL TM_USE_LINE(grid_line(idim,impl_grid))
	      ENDIF
 70	   CONTINUE

	ENDIF

* catalog implicit definition (which is temporarily stored on the grid
* stack) as a dynamic grid -- checking for existing matching grids
	IF ( cx_has_impl_grid(cx) ) THEN
	   CALL TM_GET_LIKE_DYN_GRID(impl_grid,its_dyn,use_grid,
     .			grd_stk_ptr, status)
	   IF (status .NE. merr_ok) GOTO 5000
	   IF ( its_dyn ) THEN
	      IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('allocate', isact_class_impl_grid, use_grid, 0)
	   ELSE
	      cx_has_impl_grid(cx) = .FALSE.
	   ENDIF

* ... deallocate the (now unnecessary) implicit stack-based grid
#ifdef debug_dyn_grids
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .	  ('dealloc ', isact_class_impl_grid, impl_grid, 0)
#endif
	   CALL DEALLO_GRID(status)

	   cx_grid(cx) = use_grid
	ENDIF
cc what is dsg_xlate_grid(dset) 

* flesh out regridding transformations
	DO 100 idim = 1, nferdims
	   use_axis = grid_line( idim, use_grid )
	   std_axis = grid_line( idim, std_grid )

* ... this only means that it *may be* an implied "@FMRC"
	   at_fmrc = (idim.EQ.t_dim .OR. idim.EQ.f_dim)
     .             .AND. cx_aux_var(idim,cx) .NE. unspecified_int4

* The 'regrid-to-self' transformations will operate even on the same axis
* Otherwise mark as prgrd_exact
	   IF ( use_axis .EQ. std_axis ) THEN
	      IF (REGRID_TO_SELF (cx_regrid_trans(idim, cx) ))   THEN
c	         
	      ELSE
	         cx_regrid_trans( idim, cx ) = prgrd_exact
	      ENDIF
	   ELSEIF ( cx_regrid_trans(idim,cx) .EQ. unspecified_int4 ) THEN
	      IF (cx_naux(cx) .EQ. 0) THEN
	         cx_regrid_trans( idim, cx ) = prgrd_linear_interp
	      ELSEIF ( at_fmrc 
     .		.AND. cx_aux_var(t_dim,cx) .EQ. cx_aux_var(f_dim,cx) ) THEN
	         cx_regrid_trans( idim, cx ) = pauxrgrd_fmrc ! T-F plane 
	      ELSE
! 3/2016 - "@FMRC" is implied whenever aux T or F regrid with 2dtime as aux var
	         IF (at_fmrc)
     .              at_fmrc = (cx_aux_cat(idim,cx) .EQ. cat_file_var)
     .                   .OR. (cx_aux_cat(idim,cx) .EQ. cat_agg_f_var)
* short-circuit: only call the function if it is one of this above two types
	         IF ( ITS_FMRC_2DTIME(dset, cx_aux_var(idim,cx)) ) THEN
	            cx_regrid_trans( idim, cx ) = pauxrgrd_fmrc
	         ELSE
	            cx_regrid_trans( idim, cx ) = pauxrgrd_linear
	         ENDIF
	      ENDIF
	   ENDIF
 100	CONTINUE

* change X and Y linear average to area average if appropriate  ! 8/92
        IF ( cx_regrid_trans(x_dim,cx) .EQ. prgrd_average
     . .AND. cx_regrid_trans(y_dim,cx) .EQ. prgrd_average ) THEN
           IF ( GEOG_COS_FACTOR(y_dim,std_grid)
     .    .AND. GEOG_COS_FACTOR(y_dim,use_grid) ) THEN
              cx_regrid_trans(x_dim,cx) = prgrd_xy_ave
              cx_regrid_trans(y_dim,cx) = prgrd_xy_ave
           ENDIF
        ENDIF

* change X and Y linear interpolation to call do_xylin_regrid if appropriate  ! 6/11
        IF ( cx_regrid_trans(x_dim,cx) .EQ. prgrd_linear_interp
     . .AND. cx_regrid_trans(y_dim,cx) .EQ. prgrd_linear_interp ) THEN
           IF ( GEOG_COS_FACTOR(y_dim,std_grid)
     .    .AND. GEOG_COS_FACTOR(y_dim,use_grid) ) THEN
              cx_regrid_trans(x_dim,cx) = prgrd_xy_lin
              cx_regrid_trans(y_dim,cx) = prgrd_xy_lin
           ENDIF
        ENDIF

 5000	IF (made_aux_line_dummy) CALL TM_DEALLO_DYN_LINE(src_line) ! is this needed?
	RETURN

* error exits
! 5100	CALL ERRMSG( ferr_invalid_command, status,
!     .		'mixed regridding and delta limits on '//ww_dim_name(idim)
!     .		//' axis of '//VAR_CODE(cx_category(cx),cx_variable(cx)),
!     .								*5000)
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'delta non-positive on '//ww_dim_name(idim)
     .		//' axis of '//VAR_CODE(cx_category(cx),cx_variable(cx)),
     .								*5000)
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'outside limits: '//ww_dim_name(idim)
     .		//' axis of '//VAR_CODE(cx_category(cx),cx_variable(cx)),
     .								*5000)
 5400	CALL ERRMSG( ferr_invalid_command, status,
     .		'Cannot use dates - not a calendar axis: '//
     .		VAR_CODE(cx_category(cx),cx_variable(cx)),
     .								*5000)
 5410	CALL ERRMSG( ferr_trans_nest, status,
     .               'Can only transform the '//ww_dim_name(orientation)
     .		      //' axis of DSG variable '//
     .		     VAR_CODE(cx_category(cx),cx_variable(cx)), *5000 )
 5420	CALL ERRMSG( ferr_trans_nest, status,
     .               'Can only transform the E '
     .		      //' axis of DSG variable '//
     .		     VAR_CODE(cx_category(cx),cx_variable(cx)), *5000 )
 5430	CALL ERRMSG( ferr_trans_nest, status,
     .         'Source grid has no axis along DSG feature orientation '
     .		     //VAR_CODE(cx_category(cx),cx_variable(cx)), *5000 )
	END
