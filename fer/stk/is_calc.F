	SUBROUTINE IS_CALC ( memory, *, *, status )

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
* disk but which can be computed from disk data by model-diagnostic code
* ( eg computing heat advection from temperature and velocity fields )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/17/87
* V200:  7/24/89 - 4D symmetric version based on ISP stack

* calling argument declarations:
	INTEGER	status
	REAL	memory(*)
* normal exit is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level
* V312: 5/94 - array "memory" as a calling argument

* internal variable declarations:
	LOGICAL	dummy_next, special_var, k_eq1
	INTEGER hi_cx, lo_cx, lo_isp, cvar, nitem, item0, item, comp,
     .		next_var, dset

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/instance_table.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcalc_vars.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INSTANCE_TABLE.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of next activity ( component to get )
*	 isp	<--	... ( current stack level )
*	isp-1		...
*	isp-2	<--	...	components of this cvar on stack
*	isp-n		...
*	lo_isp	<--	level controlling evaluation of cvar

* --- end of introductory code ---

* test to see if first time called - designate stack activity as algebra
	IF ( is_act(isp) .EQ. isact_request ) THEN
	   lo_isp = isp
	   is_phase( lo_isp ) = 0
	   is_act  ( lo_isp ) = isact_calculate
	ELSE
	   isp = isp + 1			! point to component just got
	   lo_isp = is_obj( isp )
	   hi_cx  = is_cx( isp )
	ENDIF

* convenience values
	lo_cx  = is_cx( lo_isp )
	cvar   = cx_variable( lo_cx )
	dset   = cx_data_set( lo_cx )
	nitem  = cvar_frame_length( cvar )
	item0  = cvar_frame_start( cvar )

* special needs for components of some variables
	IF (cvar .GT. pcv_spec_flg) cvar = cvar - pcv_spec_flg
* ... variables always requiring k=1 as lower Z limit
	k_eq1 = cvar.EQ.pdpdx  .OR. cvar.EQ.pdpdy
     .	   .OR. cvar.EQ.pw_u   .OR. cvar.EQ.pw_t

* initialize stacks
	IF ( is_phase( lo_isp ) .EQ. 0 ) THEN
	   IF ( mode_diagnostic )
     .			CALL DIAGNOSTIC_OUT( 'compute', lo_cx, point_to_cx )
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   hi_cx = cx_stack_ptr
	ENDIF

* set up to get next component
 100	is_phase(lo_isp) = is_phase(lo_isp) + 1
	item = is_phase(lo_isp)			! points into cvar table
	IF ( item .GT. nitem ) GOTO 1000
	comp = item + item0

* unravel the variable coding of the next component
	next_var = cv_variable( comp )
	special_var = next_var .GT. pcv_spec_flg
	IF ( special_var ) next_var = next_var - pcv_spec_flg

* the next component may be a dummy place holder
	IF ( special_var ) THEN
* this component was flagged for data-set-dependent special consideration
* ... different action depending on what variable needs this component
	   IF ( cvar .EQ. pqrad ) THEN
	      dummy_next = dp_phil_qflux( dset )	! need SST ?
	   ELSEIF ( cvar .EQ. pair ) THEN
	      dummy_next = dp_const_air_sst( dset )	! need CAIR ?
     .			.AND. cx_hi_ss(lo_cx,y_dim) .LT. 91
	   ELSEIF ( cvar .EQ. pcair ) THEN
	      dummy_next = .FALSE.
	   ELSE
	      CALL ERRMSG
     .			( ferr_internal, status, 'cv_spec_flg', *5000 )
	   ENDIF
	ELSEIF ( cv_lo_d3(comp) .GT. pcv_keq1_flg ) THEN
* component needed only at surface calculations
* ... are we at the surface ?
	   dummy_next = cx_lo_ss( lo_cx,z_dim ) .NE. 1
	ELSE
	   dummy_next = .FALSE.
	ENDIF

* process dummy place holder
	IF ( dummy_next ) THEN
	   isp = isp + 1
	   is_mr(isp) = dummy_mr
	   is_obj( isp ) = lo_isp	! claimed by this cvar calculation
	   GOTO 100
	ENDIF

* create context for the next component needed
	CALL CVAR_CONTEXT( lo_cx, hi_cx, comp, k_eq1, status )
	IF ( status .NE. ferr_ok ) RETURN

* request this variable on the interpretation stack ( via RETURN 1 )
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx ( isp ) = hi_cx
	is_obj( isp ) = lo_isp		! claimed by this cvar calculation
	IF ( cx_category(hi_cx) .EQ. cat_user_var ) THEN
* ... special action if this component variable is user-defined (uvar)
	   is_uvar( isp ) = cx_variable( hi_cx )
	ENDIF
	RETURN 1

* all components are ready - do the calculation
 1000	CALL CALC_CVAR( memory,
     .		        lo_cx, is_mr(lo_isp+1), is_mr(lo_isp), status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* pop off the stack space used by the components
	isp = isp - nitem

* pop off context used for components
	IF ( hi_cx .NE. cx_stack_ptr ) WRITE (6, *) 'is_calc_stk' ! temp diag
	cx_stack_ptr = cx_stack_ptr - 1
	RETURN 2

* error exit
 5000	RETURN
	END	
