	SUBROUTINE GET_CONTEXT_GRID ( cx, status )

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

* $Id$ 
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
#ifdef unix
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'interp_stack.parm'
	include 'xcontext.cmn'
	include 'xalgebra.cmn'
	include 'xtext_info.cmn'
	include 'xprog_state.cmn'
	include 'implicit.parm'
	include 'tmap_dims.parm'
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xfr_grid.cmn'   ! stack ptr should be in TMAP lib
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'TMAP_FORMAT:IMPLICIT.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'   ! stack ptr should be in TMAP lib
#endif

* calling argument declarations:
	INTEGER	cx, status

* internal variable declarations:
        LOGICAL   GEOG_COS_FACTOR, GEOG_LABEL,
     .		  its_dyn, has_impl_axis
	INTEGER	  KNOWN_GRID, TM_FIND_LIKE_GRID, TM_FIND_LIKE_DYN_GRID,
     .		  use_grid, ilo, ihi, n,
     .		  cat, var, dset, std_grid, std_axis, use_axis, idim,
     .		  base_grid, impl_grid, catlg_grid, slen
	REAL*8	  SECS_TO_TSTEP, dlo, dhi, delta
	CHARACTER VAR_CODE*24

* initialize
	cat  = cx_category( cx )
	var  = cx_variable( cx )
	dset = cx_data_set( cx )
        use_grid = cx_grid( cx )

* by this point the variable and data set must be known
	IF ( var .EQ. unspecified_int4 )	CALL ERRMSG
     .			( ferr_syntax,status,'no variable specified',*5000 )
	IF ( ( cat.EQ.cat_file_var .OR. cat.EQ.cat_calc_var )
     .	.AND. dset.EQ.unspecified_int4 )	CALL ERRMSG
     .			( ferr_state_not_set,status,'SET DATA_SET',*5000 )

* determine the standard grid for this variable
	std_grid = KNOWN_GRID( dset, cat, var )

* if the standard grid isn't known then return to the stack to get it
	IF ( std_grid .EQ. unspecified_int4 ) THEN
	   IF ( cx_has_impl_grid(cx) ) THEN
#ifdef debug_dyn_grids
	      IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('dealloc ',isact_class_impl_grid,cx_impl_grid(cx),0)
#endif
	      CALL DEALLO_GRID(status)
	   ENDIF
	   RETURN
	ENDIF

* regridding implied through delta values or special transformations?
	has_impl_axis = .FALSE.
	DO 10 idim = 1, 4
	   IF (cx_delta(idim,cx) .NE. unspecified_val8) THEN
* ... make sure the specified delta makes sense (this test moved 1/96)
	      IF ( cx_delta(idim,cx) .NE. unspecified_val8
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
	   ENDIF
* ... make the implicit axes and save them in the implicit grid
	   DO 20 idim = 1, 4
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
	            IF ( idim .EQ. t_dim ) THEN
* 8/97 *kob* use cx_calender to check if time is date rather than time step
	               IF ( cx_calendar(cx) ) THEN
	                  IF (.NOT.GEOG_LABEL(idim,std_grid)) GOTO 5400
	                  dlo = SECS_TO_TSTEP(std_grid,-1.*cx_lo_ww(idim,cx))
	                  dhi = SECS_TO_TSTEP(std_grid,-1.*cx_hi_ww(idim,cx))
			  delta = delta * 3600
     .				/ line_tunit(grid_line(idim,std_grid)) !was hrs
	               ENDIF
	            ENDIF
	            CALL TM_GET_LIKE_DYN_LINE(	idim,
     .						dlo,
     .						dhi,
     .						delta,
     .						grid_line(idim,std_grid),
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
	IF ( .NOT. cx_unstand_grid( cx ) ) THEN
	   cx_grid( cx ) = std_grid
	   RETURN
	ENDIF

* handle implicit (dynamic) grid creation
	IF ( cx_has_impl_grid(cx) ) THEN
* ... build a (temporary) implicit grid
	   IF ( use_grid .NE. unspecified_int4 ) THEN
	      base_grid = use_grid
	   ELSE
	      base_grid = std_grid
	   ENDIF
	   impl_grid = cx_impl_grid(cx)
* ... any axis that wasn't given by the user comes from the source grid
	   DO 50 idim = 1, 4
	      IF ( grid_line(idim,impl_grid).EQ.unspecified_int4 )
     .		   grid_line(idim,impl_grid) = grid_line(idim,base_grid)
 50	   CONTINUE
	   use_grid = impl_grid

* increment the use count of dynamic axes that are in-use by the temporary grid
	   DO 60 idim = 1, 4
 60	   IF (grid_line(idim,impl_grid) .GT. max_lines) CALL
     .				TM_USE_LINE(grid_line(idim,impl_grid))
	ENDIF

* 9/95 check for incompatibilities in grid geometry -- requiring implicit grid
* (i.e. chk for a normal axis in one grid without same in other)
	DO 70 idim = 1, 4
	   IF (  (grid_line(idim,std_grid).EQ.mnormal)
     .    .NEQV. (grid_line(idim,use_grid).EQ.mnormal) )   THEN
* ... allocate and initialize some work space if we don't already have it
	      IF (.NOT.cx_has_impl_grid(cx)) THEN 
	         CALL ALLO_GRID( impl_grid,status )
	         IF (status .NE. ferr_ok) GOTO 5000
	         CALL TM_COPY_GRID_W_LINE_USE(use_grid, impl_grid)
	         grid_name(impl_grid) = '(conformable)'
	         cx_has_impl_grid(cx) = .TRUE.
!	         cx_impl_grid(cx) = impl_grid
#ifdef debug_dyn_grids
                 IF ( mode_diagnostic )  CALL DIAG_OP
     .              ('allocate', isact_class_impl_grid, impl_grid, 0)
#endif
	      ENDIF
	      IF (grid_line(idim,impl_grid) .GT. max_lines) CALL
     .				TM_DEALLO_DYN_LINE(grid_line(idim,impl_grid))
	      grid_line(idim,impl_grid) = grid_line(idim,std_grid)
	      IF (grid_line(idim,impl_grid) .GT. max_lines) CALL
     .				TM_USE_LINE(grid_line(idim,impl_grid))
	   ENDIF
 70	CONTINUE

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

* flesh out regridding transformations
	DO 100 idim = 1, 4
	   use_axis = grid_line( idim, use_grid )
	   std_axis = grid_line( idim, std_grid )

	   IF ( use_axis .EQ. std_axis ) THEN
	      cx_regrid_trans( idim, cx ) = prgrd_exact
	   ELSEIF ( cx_regrid_trans(idim,cx) .EQ. unspecified_int4 ) THEN
	      cx_regrid_trans( idim, cx ) = prgrd_linear_interp	      
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

 5000	RETURN

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
	END
