      SUBROUTINE IS_ALGEBRA( memory, *, *, isp_base, status )

* evaluate an algebraic evpression component by component on the 
* interpretation stack.  When component variables are required request them
* by return to the top of the stack processor in "request" mode (RETURN 1).

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/12/87
* revision 0.1 - 6/ 3/87 - corrected 'alg_stk_err' in "LIST 1"
* revision 0.2 - 7/28/87 - changed method of constant evaluation to be like
*			   vars and pseudo-vars in order to avoid stack errors
*			   from constants left on stack when this subr. exits
* revision 0.2a- 8/29/87 - context stack bug for norm trans of constant
* V200:  6/20/89 - 4D symmetric version, numerous changes
*	11/29/89 - include constants and pseudo-variables on context merges
* V312:  5/94 - array "memory" as a calling argument
* V420:  9/95 - deallocate dynamic grid (if any) at cleanup time
* V450:  1/97 - process alg_child_var and alg_grid_chg_fcn
*        7/97 - generate child var name using GCF_PARENT_UVAR
* v491   3/98 - sgi benchmarking uncovered bug in psuedo_variable processing - 
*               no valid "name" was being set up   *kob/sh*
*        4/98 - pass default context (lo_cx) to MERGE_CONTEXT *jc*
*

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	isp_base, status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal return is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

* internal variable declarations:
	LOGICAL	EXPLICIT_UVAR, ITSA_GCVAR, its_gc
	INTEGER KNOWN_GRID, TM_LENSTR1, GCF_PARENT_UVAR,
     .		hi_mr, hi_cx, lo_cx, big_cx, dflt_cx,
     .		operator, start, end,
     .		i, dset, lo_isp, uvar, item, itype, mr_constant, idim,
     .		nitem, grid, new_mr, len_name
	REAL	constant
	CHARACTER const_buff*20, name*128

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of next activity ( component to get )
*	 isp	<--	... ( current stack level )
*	isp-1		...
*	isp-2	<--	...	components of this uvar on stack
*	isp-n		...
*	lo_isp	<--	level controlling evaluation of uvar

* --- end of introductory code ---

* test to see if first time called - designate stack activity as algebra
	IF ( is_act(isp) .EQ. isact_request ) THEN
	   lo_isp = isp
	   is_phase( lo_isp ) = 0
	   is_act  ( lo_isp ) = isact_algebra
	ELSE
	   isp = isp + 1			! point to component just got
	   lo_isp = is_obj( isp )
	   hi_cx  = is_cx( isp )
	ENDIF

* convenience values
	lo_cx  = is_cx( lo_isp )
	big_cx = is_big_cx( lo_isp )
	uvar   = is_uvar( lo_isp )
	nitem  = uvar_num_items( uvar )
	dset   = cx_data_set( lo_cx )

* is it a grid-changing variable?
	its_gc = ITSA_GCVAR(uvar)

* create, initialize or merge overall context for the algebraic expression
	IF ( is_phase(lo_isp) .EQ. 0 ) THEN
	   cx_grid( lo_cx ) = KNOWN_GRID( dset ,cat_user_var, uvar )
	   IF ( mode_diagnostic )
     .			CALL DIAGNOSTIC_OUT( 'eval   ', lo_cx, point_to_cx )
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   big_cx = cx_stack_ptr
	   is_big_cx( lo_isp ) = big_cx
	   is_sum   ( lo_isp ) = isalg_transfer
!	ELSEIF ( cx_category( hi_cx ) .EQ. cat_constant
!     .	   .OR. ( cx_category(hi_cx) .EQ. cat_pseudo_var
!     .		  .AND. .NOT.cx_unstand_grid(hi_cx)      )  ) THEN
!	   GOTO 100		! can't construct a context from these

* 1/97 - need a somewhat separate path through the merging of contexts when
*	dealing with grid changing variables. For non-grid-changing vars
*	we may have "LIST A", inwhich case the merged context needs to look
*	just like "A". For grid-changing variables this can never occur, since
*	they are always of the form GCFUNC(arg1, arg2, ...), but it is not
*	safe to begin the merging with TRANSFER_CONTEXT because the first
*	arg may be designated not to contribute all of its axes to the merged
*	result 
	ELSEIF ( its_gc ) THEN
	   IF ( is_sum(lo_isp) .EQ. isalg_transfer ) THEN
	      CALL INIT_EMPTY_CONTEXT( big_cx)
	      CALL MERGIFY_CX( lo_cx, uvar, big_cx, status )
	      is_sum( lo_isp ) = isalg_merge_1
	   ELSEIF ( is_sum(lo_isp) .EQ. isalg_merge_1 ) THEN
* ... this allows us to distinguish between single component and
*	multi-component expressions. "LIST A" should document "A", not "E315"
	      is_sum( lo_isp ) = isalg_merge_n
	   ENDIF
	   item = is_phase(lo_isp)			! points into uvar_text
	   itype = uvar_item_type ( item, uvar )
	   start = uvar_item_start( item, uvar )
	   end   = uvar_item_end  ( item, uvar )
	   CALL MERGE_CONTEXT(	its_gc, item, lo_cx, big_cx, hi_cx,
     .				uvar_text(uvar)(start:end), status )
	   IF ( status .NE. ferr_ok ) RETURN

* pre grid-changing code - 1st arg treated differently from others
	ELSEIF ( is_sum(lo_isp) .EQ. isalg_transfer ) THEN
	   CALL TRANSFER_CONTEXT( hi_cx, big_cx )
	   is_sum( lo_isp ) = isalg_merge_1
	ELSE
	   IF (  is_sum(lo_isp) .EQ. isalg_merge_1 ) THEN
	      CALL MERGIFY_CX( lo_cx, uvar, big_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      is_sum( lo_isp ) = isalg_merge_n
	   ENDIF
	   item = is_phase(lo_isp)			! points into uvar_text
	   itype = uvar_item_type ( item, uvar )
	   start = uvar_item_start( item, uvar )
	   end   = uvar_item_end  ( item, uvar )
	   CALL MERGE_CONTEXT( its_gc, item, lo_cx, big_cx, hi_cx,
     .				uvar_text(uvar)(start:end), status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* ****** (continue) LOOP through expression ***********************************
* process the next item in the expression from the uvar buffer
 100	is_phase(lo_isp) = is_phase(lo_isp) + 1
	item = is_phase(lo_isp)			! points into uvar_text
	IF ( item .GT. nitem ) GOTO 1000
	itype = uvar_item_type ( item, uvar )
	start = uvar_item_start( item, uvar )
	end   = uvar_item_end  ( item, uvar )
	IF     ( itype .EQ. alg_constant   ) THEN
	   GOTO 200
	ELSEIF ( itype .EQ. alg_child_var   ) THEN
	   GOTO 300
	ELSEIF ( itype .EQ. alg_variable   ) THEN
	   GOTO 305
	ELSEIF ( itype .EQ. alg_pseudo_var ) THEN
	   GOTO 400
	ELSEIF ( itype .EQ. alg_function
     .	    .OR. itype .EQ. alg_operator
     .	    .OR. itype .EQ. alg_log_struct   ) THEN
	   GOTO 500
	ELSEIF ( itype .EQ. alg_grid_chg_fcn ) THEN
	   GOTO 600
	ELSEIF ( itype .EQ. alg_no_op ) THEN
	   GOTO 100
	ELSE
	   STOP 'unknown algebraic category'
	ENDIF

* constant - with possible transformations (eg. integrated --> 1[Z=1:100@DIN])
* ... allocate a context buffer 
 200	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
	CALL CONSTANT_CONTEXT   ( uvar_text(uvar)(start:end),
     .				  lo_cx,
     .				  hi_cx,
     .				  status )
	IF ( status .NE. ferr_ok ) RETURN
* ... request this constant on the interpretation stack ( via RETURN 1 )
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx ( isp ) = hi_cx
	is_obj( isp ) = lo_isp	! claimed by this level
	RETURN 1

* child variable: treat it just like a regular variable except that 
* its name is not found in the text. Instead generate its name from
* its position and its parent variable
* example: for the definition	LET A = 5 * GCFUNC(b,c)
*    the string "GCFUNC(b,c)" is processed as a child variable
 300	CALL GCF_CHILD_VAR_NAME( GCF_PARENT_UVAR(uvar), start, name )
	len_name = TM_LENSTR1(name)
	GOTO 310

* variable: parse name, update full expression context and evaluate
* ... allocate a context buffer for components
 305	name = uvar_text(uvar)(start:end)
	len_name = end - start + 1
 310	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
* ... for GC variables the default context may get modified before using it
*     to get an argument (e.g. for a fcn equivalent to x=@sbx:5 we'd have
*     LIST/I=5:10 XSBX5(b) requiring I=3:12 of argument "b".
	IF (its_gc ) THEN
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   dflt_cx = cx_stack_ptr
	   CALL GCF_IMPOSE_ARG_LIM_DFLTS( name(:len_name), item,
     .					  lo_cx, dflt_cx, uvar, status )
	   IF (status .NE. ferr_ok) RETURN
	ELSE
	   dflt_cx = lo_cx	! for non-GC variables
	ENDIF
	CALL GET_VAR_CONTEXT (  name(:len_name),
     .				dflt_cx,
     .				hi_cx,
     .				status )
	IF ( status .NE. ferr_ok ) RETURN
	IF (its_gc ) cx_stack_ptr = cx_stack_ptr - 1
* ... request this variable on the interpretation stack ( via RETURN 1 )
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx ( isp ) = hi_cx
	is_obj( isp ) = lo_isp	! claimed by this level
	IF ( cx_category(hi_cx) .EQ. cat_user_var ) THEN
* ... special action if this component variable is user-defined (uvar)
	   is_uvar( isp ) = cx_variable( hi_cx )
	ENDIF
	RETURN 1

* pseudo-variable: parse mods, update full expression context and evaluate
* ... allocate a context buffer for components
 400	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
	IF (its_gc ) THEN	! see explanation under "variable' (stmt 310)
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   dflt_cx = cx_stack_ptr
* use uvar_text as name to pass to gcf_impose...*kob* 3/98
	   CALL GCF_IMPOSE_ARG_LIM_DFLTS( uvar_text(uvar)(start:end), item,
     .					  lo_cx, dflt_cx, uvar, status )
	   IF (status .NE. ferr_ok) RETURN
	ELSE
	   dflt_cx = lo_cx	! for non-GC variables
	ENDIF
	CALL PSEUDO_VAR_CONTEXT ( uvar_text(uvar)(start:end),
     .				  dflt_cx,
     .				  hi_cx,
     .				  idim,
     .				  status )
	IF ( status .NE. ferr_ok ) RETURN
	IF (its_gc ) cx_stack_ptr = cx_stack_ptr - 1
	IF ( cx_lo_ss(hi_cx,idim) .EQ. unspecified_int4 )  CALL ERRMSG
     .		(ferr_dim_underspec, status,uvar_text(uvar)(start:end), *5000)
! temp diag
	IF ( cx_grid(hi_cx) .EQ. unspecified_int4 ) CALL ERRMSG
     .			( ferr_internal, status, 'isalg_ps', *5000 )
* ... request this variable on the interpretation stack ( via RETURN 1 )
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx ( isp ) = hi_cx
	is_obj( isp ) = lo_isp	! claimed by this level
	RETURN 1

* function logic structure or operator:
* ...compute new value replacing component(s) (if any) on stack
* ...and changing variable protections appropriately
* ...also pop the space they used on the context stack
 500	CALL IS_DO_OP( memory, itype, uvar_text(uvar)(start:end),
     .		       lo_isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	GOTO 100

* grid-changing function 
* ...(internal or external GC function)
* ...in addition to the manipulations of IS_DO_OP above, this procedure
* ...has the additional flexibility that the result grid need not be
* ...inferrable from the input component grids
 600	CALL IS_DO_GC_FCN( memory, itype, uvar_text(uvar)(start:end),
     .		       lo_isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	GOTO 100

* DONE - clean up
 1000	hi_cx = is_cx( isp )
	hi_mr = is_mr( isp )
	is_mr( lo_isp ) = hi_mr
	isp = isp - 1
	IF ( isp .NE. lo_isp ) 
     .	   CALL ERRMSG( ferr_syntax, status, 'RPN', *5000 )

* Deallocate dynamic grid if one is associated with this result
* This will (I think) only happen if a single component (no IS_DO_OP call)
* result with a dynamic grid is requested
* (This will not actually dismiss the grid -- it should always merely decrement
*  the use count)
	IF ( cx_has_impl_grid(hi_cx) ) THEN
	   CALL DEALLO_DYN_GRID(cx_grid(hi_cx))
	   cx_has_impl_grid(hi_cx) = .FALSE.
	ENDIF

* check for incomplete merged context
* this is complicated in order to have implicit user-defn'd vars like
* "LIST TEMP" get cataloged under "TEMP"
* whereas "LIST TEMP/10" gets cataloged under the implied user-defn'd var name
* and whereas the explicit "DEFINE A=TEMP" gets cataloged under "A".
* Use the isp level to determine if it's implicit - implicit always at base lev
* MERGIFY_CX "claims" a context under the user-defn'd name.
	IF ( is_sum(lo_isp) .EQ. isalg_transfer ) THEN
* ... expression entirely of unmodified pseudo-vars and constants
	   CALL TRANSFER_CONTEXT( hi_cx, big_cx )
	   IF ( cx_category(big_cx) .EQ. cat_temp_var
     .	   .OR. EXPLICIT_UVAR(uvar) ) THEN
* ... expr. like "LIST I+J" or "LIST A" after "DEFINE VAR A=I"
	      CALL MERGIFY_CX( lo_cx, uvar, big_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ELSEIF ( is_sum(lo_isp).EQ.isalg_merge_1
     .	.AND.  ( nitem.GT.1 .OR. EXPLICIT_UVAR(uvar) )  ) THEN 
* ... expression with constants, pseudo-variables and a single variable
* ... like "LIST U/10" or single variable explicit uvars like "DEFINE NEW=U"
	   CALL MERGIFY_CX( lo_cx, uvar, big_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* finalize the result on the stack - it may be the result of some algebra or
* a simple variable or a constant or pseudo-variable
* "temporary variables" are the result of algebraic operations
* ... re-catalog them under their user-variable names
* explicit user-defined variables (e.g. "DEFINE VAR NEW=X"; "LIST NEW") are
* always re-cataloged under the user-defined name
* Merged context info is then passed back to the calling context.
* implicit uvars that are single pseudo-variables or constants are left as-is
* ... to be deleted as soon as used (e.g. "LIST X")
* implicit uvars that are single variables (e.g. "LIST U") need their "in-use"
* flags removed (otherwise done when they're used as operands)
	IF ( mr_category( hi_mr ) .EQ. cat_temp_var ) THEN
* ... result of algebraic computation
	   CALL RE_ASSIGN_VARIABLE( hi_mr, big_cx )
	   CALL PASS_AMBIG_LIMS( lo_cx, hi_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSEIF ( EXPLICIT_UVAR(uvar) ) THEN
* ... result of a simple definition like DEFINE VARIABLE UAVE=U[K=@AVE]
*	or DEFINE VARIABLE NEW=X
	   IF ( mr_category(hi_mr) .NE. cat_pseudo_var
     .	  .AND. mr_category(hi_mr) .NE. cat_constant   )
     .						CALL MR_NOT_IN_USE(hi_mr)
	   CALL CREATE_MEM_VAR( big_cx, new_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( lo_isp ) = new_mr
	   CALL COPY_GRID( memory(1,mr_blk1(hi_mr)),  hi_mr,
     .			   memory(1,mr_blk1(new_mr)), new_mr )
	   CALL PASS_AMBIG_LIMS( lo_cx, new_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   IF ( mr_category(hi_mr) .NE. cat_pseudo_var
     .	  .AND. mr_category(hi_mr) .NE. cat_constant   )
     .						CALL MR_NOT_IN_USE(hi_mr)
	   CALL TRANSFER_CONTEXT( big_cx, lo_cx )
	ENDIF

* pop off context for the last component received and "big" (merged) context
	IF ( big_cx .NE. cx_stack_ptr-1
     .	.OR. big_cx .NE. lo_cx+1 ) WRITE (6, *) 'is_alg_2' ! temp diag
	cx_stack_ptr = cx_stack_ptr - 2
	RETURN 2

* error exit
 5000	RETURN
	END	
