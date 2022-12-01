	SUBROUTINE PSEUDO_VAR_CONTEXT
     .		      ( full_pvar_name, 
     .			dflt_cx,
     .			pvar_cx,
     .			pvdim,
     .			status )

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
* given a pseudo-variable
* return in pvar_cx context information describing the exact instance
* including grid and region

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* revision 0.1 - 5/5/88  - FINISH_CONTEXT_MODS --> FLESH_OUT_CONTEXT
*			   added CONFINE_CONTEXT
* V200: 6/26/89
* V420: 9/27/95 - allow implicit grids underlying pseudo-var
*	10/95	- allow implicit axes like i[i=1:10:2]
*	11/95   - code to deal with T[T=date:date:delta]
* 
* kob  8/25/97 - use newly created cx_calender to check whether time is a date or time_step
* V500: 2/22/99 *sh* - if ABSTRACT grid or normal axis then use XABSTRACT (or
*			YABSTRACT, ZABSTRACT, TABSTRACT)
*		       see Note at "the_real-ABSTRACT"
* V510: 8/99 *sh* - for g=user_var the returned grid is LT unspecified_int4
*		... having the text position of uvar encoded in it
*		... exit out earliy in this case
*	12/99 *sh* - for formatted lat or lon (e.g. Y[y=0:20n:2]), set units
*	 4/00 *sh* - all grids and axes now maintain use counts
* V530:  8/00 *sh* - assert float data type for pseudo-variable
* 1/03 - *kob* g77 port - Need to use risc_buff for arbitrary concatenations
* V581*acm*  6/05 - For fix to bug 1271, add flag line_shift_origin
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V720  *acm* 2/17 ticket 2511 Dynamic grids on user-defined variables
* V720 *sh* 5/17 DSG pseudo-variables -- depend on grid, not axis
* V74+ 7/18 *acm* When rowsize variable was synthesized, it's marked with a negative var-id
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include 'tmap_dims.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include	'xfr_grid.cmn'
	include	'xprog_state.cmn'
	include 'xrisc.cmn'
	include 'tmap_errors.parm'
	include 'xunits.cmn_text'
	include 'implicit.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
# include "tmap_dset.parm"

* calling argument declarations:
	INTEGER		dflt_cx, pvar_cx, pvdim, status
	CHARACTER*(*)	full_pvar_name

* internal variable declarations:
	LOGICAL		TM_ITSA_DSG, its_dyn, has_impl_axis, calendar,
     .			alloed_tmp, its_dsg, has_coords
	INTEGER		CGRID_AXIS, TM_DSG_DSET_FROM_GRID,
     .			mods_cx, idim, pvax, base_grid,
     .			impl_grid, like, tmp, the_real_ABSTRACT, dset,
     .			coord_var, pvtype

* internal parameter declarations
	REAL*8		daysinsecs
	PARAMETER     ( daysinsecs = 60.*60.*24.)

* initialize fixed length string for RISC
        risc_buff = full_pvar_name
        len_rbuff = MIN( size_rbuff, LEN(full_pvar_name) )
* ... Note: the variable mgrid_abstract gets changed by SET GRID, so here
*     we use an ugly hack to capture the grid that is actually named "ABSTRACT"
	the_real_ABSTRACT = mgrid_xabstract - 2

* use default context as a starting point
	CALL TRANSFER_CONTEXT ( dflt_cx, pvar_cx )

* find a context slot for the modifications that may be attached to the var name
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* get variable and attached modifications
	CALL PARSE_VAR_NAME ( full_pvar_name, pvar_cx, mods_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	cx_type( pvar_cx ) = ptype_float 

* axis of pseudo_var
	pvdim = MOD( cx_variable(pvar_cx)-1, nferdims ) + 1

* temporary bug check: may not transform pseudo-variable
	IF ( cx_trans(pvdim,pvar_cx) .NE. trans_no_transform ) CALL ERRMSG
     .		( ferr_syntax, status,
     .		'cannot transform pseudo-variables: '//
     .		 risc_buff(:len_rbuff)//pCR//
     .		'use DEFINE VARIABLE (e.g. DEF VAR ii=i) and transform that', 
     .								*5000 )

* pseudo var may not depend on any data set
	IF ( .NOT.(cx_dset_gvn(mods_cx) .OR. cx_grid_needs_dset) ) THEN
	   cx_data_set(pvar_cx) = pdset_irrelevant
	ENDIF

* bug check
	IF ( cx_category(pvar_cx) .NE. cat_pseudo_var ) CALL ERRMSG
     .			( ferr_internal, status, 'pseud_v', *5000 )

* is this a DSG pseudo-variable? 5/17
	base_grid = cx_grid(pvar_cx)
	IF ( base_grid .GT. unspecified_int4) THEN
	   its_dsg = TM_ITSA_DSG(base_grid)
	ELSE
	   its_dsg = .FALSE.   ! for now, anyway
	ENDIF

	IF (its_dsg .AND. pvdim.NE.f_dim) THEN
* ... type of pseudo-var  1=IJKLMN, 2=XYZTEF, 3=BOX, 4=BOXHI, 5=BOXLO
	   pvtype = (cx_variable(pvar_cx)-1) / nferdims  + 1
	   IF (pvtype .GE.3) GOTO 5300
* ... choose the grid (instance or obs) of the indicated coordinate var
	   dset = TM_DSG_DSET_FROM_GRID(base_grid)
	   IF (pvdim .EQ. e_dim) THEN
	      coord_var = ABS(dsg_row_size_var(dset))
	   ELSE
	      coord_var = dsg_coord_var(pvdim,dset)
	   ENDIF
	   has_coords = coord_var .NE. int4_init
	   IF (has_coords) THEN
	      cx_grid(pvar_cx) = ds_grid_number(coord_var)
	   ELSE
* ... no DSG coor var exists, so treat this as an abstract axis
	      cx_grid(pvar_cx) = mgrid_xabstract + pvdim - 1  ! ?? ok to do ??
	   ENDIF      
	ELSE
	   its_dsg = .FALSE.
	ENDIF

* implicit axis of the form I[I=1:10:2] used?
	has_impl_axis = cx_delta(pvdim,pvar_cx) .NE. unspecified_val8
	IF ( has_impl_axis ) THEN

* ... make sure the specified delta makes sense
	   IF ( cx_delta(pvdim,pvar_cx) .LE. 0.0D0 ) GOTO 5200

* ... need a temporary grid to hold the axes so make it if we don't have one
	   IF ( .NOT.cx_has_impl_grid(pvar_cx) ) THEN
	      CALL ALLO_GRID_TO_CX( pvar_cx, impl_grid, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	   ENDIF
	ENDIF

* process implicit (dynamic) grids (9/95)
* complete the implicit grid definition from the pieces provided
* There is no reason that we **have to** retain any axes other than the
* needed one but it reduces the proliferation of distinct grids if we do
	IF ( cx_has_impl_grid(pvar_cx) ) THEN
* ... build the grid
*     Note: there is an (imperfect) subtlety in the selection of "base_grid"
*	In an expression like "temp+x" the pseudo-variable inherits its axis
*	from the variable "temp".  So in an expression like
*	"temp[x=lo:hi:del]+X[x=lo:hi:del]" presumably the same would be true.
*	This would require that base_grid was inherited --> pulled from 
*	cx_grid(pvar_cx).  However, the inheritance only occurs after an
*	initial pass through this routine where cx_grid(pvar_cx) is unknown.
*	This means that the inherited base_grid based could vary from pass
*	to pass -- leading to, say, the delta value 2 in X[x=lo:hi:2] being
*	applied first to ABSTRACT and then to itself -- leading to a delta of
*	4 and probably a crash.  This is side-stepped by using cx_grid(mods_cx)
*	but with the result that the grid is not inherited.  To  work around
*	the user must use "temp[x=lo:hi:del]+X[g=temp,x=lo:hi:del]"

*   ticket 2511: the grid number may be a large negative number due to the hack 
*   in get_context_mods.F which encodes the start/end limits of the uvar name 
*   into the grid. When that is the case we do not want to use that as base_grid.

	   IF ( cx_grid(mods_cx) .GT. unspecified_int4 ) THEN
	      base_grid = cx_grid(mods_cx)
	   ELSE
! 2/22/99 addition
	      IF (mgrid_abstract .EQ. the_real_ABSTRACT) THEN
	         base_grid = mgrid_xabstract + pvdim - 1
	      ELSE
		 base_grid = mgrid_abstract
	      ENDIF
	   ENDIF
	   impl_grid = cx_impl_grid(pvar_cx)
	   DO idim = 1, nferdims
	      IF ( grid_line(idim,impl_grid).EQ.unspecified_int4 )
     .		   grid_line(idim,impl_grid) = grid_line(idim,base_grid)
	   ENDDO
	ENDIF

* if a dynamic axis was requested create it with the named grid or axis
* as a template and insert it into the dynamic grid 
	IF ( has_impl_axis ) THEN
* ... make the implicit axis and save it in the implicit grid
	   like = grid_line(pvdim,impl_grid)
	   IF ( cx_by_ss(pvdim,pvar_cx) ) THEN
	      CALL TM_GET_LIKE_DYN_LINE( pvdim,
     .					 DBLE(cx_lo_ss(pvar_cx,pvdim)),
     .					 DBLE(cx_hi_ss(pvar_cx,pvdim)),
     .					 cx_delta(pvdim,pvar_cx),
     .					 like,
     .					 pline_class_stride,
     .					 grid_line(pvdim,impl_grid),
     .					 status)
	      cx_lo_ss(pvar_cx,pvdim) = 1
	      cx_hi_ss(pvar_cx,pvdim) = line_dim(grid_line(pvdim,impl_grid))
	   ELSE
* ... special processing for calendar pseudo-variables and lat/long formatting
* *kob* 8/97 check cx_calendar rather then cx_lo_ww
	      calendar = (pvdim.EQ.t_dim .OR. pvdim.EQ.f_dim) .AND. 
     .		 cx_calendar(pvdim,pvar_cx)
	      alloed_tmp = .FALSE.	! assume no dyn axis allocated
	      IF ( calendar ) THEN
* ... note: a minor misbehavior:  for the pseudo-variable
* 	"T[g=temp,t=[t=1-jan-1982:1-jan-1983:24]" it would be nice to inherit
*	line_t0, etc. from g=temp.  The current method does not. (11/95)
	         cx_lo_ww(pvdim,pvar_cx) = cx_lo_ww(pvdim,pvar_cx)
     .					/(-1 * daysinsecs)	! days since BC
	         cx_hi_ww(pvdim,pvar_cx) = cx_hi_ww(pvdim,pvar_cx)
     .					/(-1 * daysinsecs)	! days since BC
* *kob* 9/997 - reset cx_calendar after conversion
		 cx_calendar(pvdim,pvar_cx) = .FALSE.
	         cx_delta(pvdim,pvar_cx) = cx_delta(pvdim,pvar_cx)/24.
	         CALL TM_ALLO_DYN_LINE( tmp, status )
	         IF ( status .NE. ferr_ok ) GOTO 5000
	         alloed_tmp = .TRUE.
	         CALL TM_COPY_LINE( like, tmp )
	         line_t0(tmp) = '1-JAN-0000'
	         line_shift_origin(tmp) = .FALSE.
	         line_units(tmp) = 'DAYS'
	         line_unit_code(tmp) = -4			! days
	         line_tunit(tmp) = daysinsecs
	         line_direction(tmp) = 'TI'
	         like = tmp
	      ELSEIF ( cx_formatted(pvdim) 
     .		.AND. (pvdim.EQ.x_dim .OR. pvdim.EQ.y_dim) ) THEN
	         CALL TM_ALLO_DYN_LINE( tmp, status )
	         IF ( status .NE. ferr_ok ) GOTO 5000
	         alloed_tmp = .TRUE.
	         CALL TM_COPY_LINE( like, tmp )
	         line_unit_code(tmp) = pun_degrees
	         IF ( pvdim .EQ. x_dim ) THEN
		    line_units(tmp) = 'DEGREES_EAST'
	            line_direction(tmp) = 'WE'
	         ELSEIF ( pvdim .EQ. y_dim) THEN
		    line_units(tmp) = 'DEGREES_NORTH'
	            line_direction(tmp) = 'SN'
	         ENDIF
	         like = tmp
	      ENDIF
	      CALL TM_GET_LIKE_DYN_LINE( pvdim,
     .					 cx_lo_ww(pvdim,pvar_cx),
     .					 cx_hi_ww(pvdim,pvar_cx),
     .					 cx_delta(pvdim,pvar_cx),
     .					 like,
     .					 pline_class_basic,
     .					 grid_line(pvdim,impl_grid),
     .					 status)
	      IF ( alloed_tmp ) THEN
	         CALL TM_DEALLO_DYN_LINE( tmp )
	      ENDIF
	   ENDIF
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   cx_delta(pvdim,pvar_cx) =  unspecified_val8
	ENDIF

	IF ( cx_has_impl_grid(pvar_cx) ) THEN
* increment the use count of dynamic axes that are in-use by the temporary grid
	   DO idim = 1, nferdims
	      CALL TM_USE_LINE(grid_line(idim,impl_grid))
	   ENDDO

* catalog the temporary grid as a dynamic grid checking for existing like grid
* ... catalog it or reuse a previously cataloged version
	   CALL TM_GET_LIKE_DYN_GRID(impl_grid,its_dyn,
     .				cx_grid(pvar_cx),grd_stk_ptr,status)
	   IF (status .NE. merr_ok) GOTO 5000
	   IF ( its_dyn ) THEN
	      IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('allocate', isact_class_impl_grid, cx_grid(pvar_cx), 0)
	   ELSE
	      cx_has_impl_grid(pvar_cx) = .FALSE.
	   ENDIF

* ... deallocate the (now unnecessary) implicit stack-based grid
#ifdef debug_dyn_grids
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .	  ('dealloc ', isact_class_impl_grid, impl_grid, 0)
#endif
	   CALL DEALLO_GRID(status)

* the dynamic (managed) grid has now replaced the temporary (stack) grid
	   impl_grid = cx_grid(pvar_cx)

	ENDIF

* unknown grid for pseudo-variable is OK when program is determining grid
* for an algebraic expression but not OK when it's time to evaluate the expr.
	IF ( cx_grid(pvar_cx) .LE. unspecified_int4 ) GOTO 500	! 8/99

* if the grid is ABSTRACT change it to X/Y/Z/TABSTRACT so other axes than
* pvdim are not effected (2/22/99)
	IF ( cx_grid(pvar_cx) .EQ. the_real_ABSTRACT )
     .		 cx_grid(pvar_cx) = mgrid_xabstract + pvdim - 1

* does the grid have a suitable axis for the pseudo-variable ?
	pvax = CGRID_AXIS( pvdim, pvar_cx )
	IF ( pvax.EQ.mnormal .AND. .NOT.its_dsg) THEN
	   cx_grid(pvar_cx) = mgrid_xabstract + pvdim - 1  ! 2/22/99
	ELSEIF (pvax.EQ. munknown ) THEN
	   GOTO 5100
	ENDIF

* insert bad data flag for variable
	CALL GET_CONTEXT_BAD_FLAG( pvar_cx )

* flesh out context removing all axes except those explicitly given
	DO idim = 1, nferdims
	   IF ( idim .NE. pvdim .AND. .NOT.its_dsg) THEN
	      CALL DEL_CX_DIM( idim, pvar_cx )
	   ELSE
	      CALL CONFINE_AXIS  ( idim, pvar_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL FLESH_OUT_AXIS( idim, pvar_cx, status ) 	! ignore status
	   ENDIF
	ENDDO

* relinquish the stack space used for the modifications
 500	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	RETURN

* error exit
 5000	RETURN
 5100	IF (cx_has_impl_grid(pvar_cx)) CALL TM_DEALLO_DYN_GRID(impl_grid)
	CALL ERRMSG( ferr_inconsist_grid, status, full_pvar_name, *5000 )
* 1/03 *kob* g77 port requires risc buff for arbitrary string concats
 5200	risc_buff = full_pvar_name
	CALL ERRMSG( ferr_invalid_command, status, 
     .	'non-positive delta value: '//risc_buff, *5000 )
 5300	CALL ERRMSG( ferr_inconsist_grid, status, 
     .              'BOX pseudovar on DSG data'//full_pvar_name, *5000 )
	END
