	SUBROUTINE IS_UVAR_GRID( *, *, status )

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
* examine an algebraic evpression component by component on the 
* interpretation stack in order to determine a defining grid for the overall
* expression.
* Detailed description of the underlying logic is in MERGE_CONTEXT
* Note: When pseudo-variables are used (eg. LIST U/Z[K=1:3]) the grids for them
*	may not be known until all other components are examined.  In this
*	case a 2nd pass is required

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/20/89
* V230:  7/21/92 - separate error message for cvar grid unknown
* V411  8/28/95 - variables defined with LET/D= need uvar_need_dset=TRUE
* V420   9-11/95 - dynamic axes and grids
*	 12/95	- set uvar_mod_pseudo if there is a pseudo-var dependency
* V450:  1/97 - process alg_child_var
*	      - and process grid-changing variables
*        7/97 - generate child var name using GCF_PARENT_UVAR
* V500: *sh* 12/98 - support for cat_const_var -- fixed arrays like "{1,3,5}"

* calling argument declarations:
	INTEGER	status
* normal return is for error conditions
* exit #1 is to request grid for a component ( stack has been pushed )
* exit #2 is to signal that the grid has been obtained and stored in uvar_grid

* internal variable declarations:
	LOGICAL	ITSA_GCVAR, its_dynamic, its_gc
	INTEGER KNOWN_GRID, TM_FIND_LIKE_GRID, TM_FIND_LIKE_DYN_GRID,
     .		TM_LENSTR1, GCF_PARENT_UVAR,
     .		 hi_cx, lo_cx, big_cx, operator, start, end, grid,
     .		i, dset, uvar, item, itype, idim, gnum, nitem, uv_grd,
     .		hi_grid, var, len_name
	CHARACTER VAR_CODE*4, LEFINT*6, name*128

#ifdef unix
	include	'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xfr_grid.cmn'
	include	'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
        INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
#endif

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level to get grids for components
*	 isp	<--	current level: evaluate user-variable grid

* --- end of introductory code ---

* initialize
	lo_cx  = is_cx( isp )
	big_cx = is_big_cx( isp )
	uvar   = is_uvar( isp )
	dset   = cx_data_set( lo_cx )
	nitem  = uvar_num_items( uvar )
	its_gc = ITSA_GCVAR(uvar)

* process the next item in the expression from the uvar buffer (last to first)
 100	is_phase(isp) = is_phase(isp) + 1
	item = nitem - is_phase(isp)	+ 1	! point into uvar_text
	IF ( item .LT. 1 ) GOTO 1000		! have processed entire def'n
	itype = uvar_item_type ( item, uvar )
	start = uvar_item_start( item, uvar )
	end   = uvar_item_end  ( item, uvar )
	IF     ( itype .EQ. alg_pseudo_var ) THEN
	   GOTO 200
	ELSEIF ( itype .EQ. alg_const_var   ) THEN
	   GOTO 250
	ELSEIF ( itype .EQ. alg_child_var   ) THEN
	   GOTO 300
	ELSEIF ( itype .EQ. alg_variable   ) THEN
	   GOTO 305
	ELSE
	   GOTO 100
	ENDIF

* pseudo-variable: merge into grid if it has explicit grid info
 200	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
	CALL PSEUDO_VAR_CONTEXT( uvar_text(uvar)(start:end),
     .				 lo_cx,
     .				 hi_cx,
     .				 idim,
     .				 status )
	IF ( status .NE. ferr_ok ) RETURN
* 12/95: flag any axes of this uvar that depend on the pseudo-variable
* (note XBOX probably doesn't need this but, hey, what the heck)
	uvar_mod_pseudo(idim,uvar) = .TRUE.
* did we get a grid for the pseudo-variable ?
	IF ( cx_grid(hi_cx) .NE. unspecified_int4 ) THEN
	   GOTO 500		! yup - merge it
	ELSE
* nope! Set flag indicating a 2nd pass is needed
	   uvar_given( idim, uvar ) = uvlim_unknown
	   cx_stack_ptr = cx_stack_ptr - 1
	ENDIF
	GOTO 100

* constant-array-variable (of type alg_const_var):
 250	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
        CALL CONST_VAR_CONTEXT ( uvar_text(uvar)(start:end),
     .                           uvar,
     .                           item,
     .                           lo_cx,
     .                           hi_cx,
     .                           status )
	IF ( status .NE. ferr_ok ) RETURN
	GOTO 500		! go merge it

* child variable: treat it just like a regular variable except that 
* its name is not found in the text. Instead generate its name from
* its position and its parent variable
* see IS_ALGEBRA for example
 300	CALL GCF_CHILD_VAR_NAME( GCF_PARENT_UVAR(uvar), start, name )
	len_name = TM_LENSTR1(name)
	GOTO 310

* variable: parse name, get full expression context
 305	name = uvar_text(uvar)(start:end)
	len_name = end-start+1
 310	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	hi_cx = cx_stack_ptr
	CALL GET_VAR_CONTEXT (  name(:len_name),
     .				lo_cx,
     .				hi_cx,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000
* do we know the defining grid for the component ?
* (not necessarily the same as the returned grid --> there may be a regrid )
	IF ( cx_category(hi_cx) .EQ. cat_user_var ) THEN
* ... 12/95: inherit the pseudo-variable flag from any component that has it
	   var = cx_variable(hi_cx)
	   DO 350 idim = 1, 4
	      IF (uvar_mod_pseudo(idim,var))
     .			uvar_mod_pseudo(idim,uvar) = .TRUE.
 350	   CONTINUE
	   uv_grd = KNOWN_GRID( cx_data_set(hi_cx), cat_user_var, var )
	   IF ( uv_grd .EQ. unspecified_int4 ) THEN
	      GOTO 900	! uvar grid not known --> get it
	   ELSE
	      GOTO 500	! uvar grid known --> merge it
	   ENDIF
	ELSEIF ( cx_grid(hi_cx) .NE. unspecified_int4 ) THEN
	   GOTO 500	! know grid --> merge it
	ELSEIF ( cx_category(hi_cx) .EQ. cat_calc_var ) THEN
	   GOTO 9100	! err: grid for cvar may not be knowable
	ELSE
	   GOTO 9000	! err: grid for non-user vars should always be known
	ENDIF

* create, initialize buffer to receive the grid
 500	IF ( is_sum(isp) .EQ. isalg_transfer ) THEN
	   grid = cx_grid(big_cx)
	   CALL INIT_EMPTY_CONTEXT( big_cx)
	   CALL INIT_GRID(grid, 'UVAR', mnormal)
	   cx_grid(big_cx) = grid
	   is_sum( isp ) = isalg_merge_n
	ENDIF

* merge the newly acquired grid with the previously merged ones
	grid = cx_grid(big_cx)
	CALL MERGE_GRID( item, its_gc, uvar_text(uvar)(start:end),
     .			 big_cx, hi_cx, uvar, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* 1/97: For GC variables, the last GC function argument has just been merged.
* Now we need to finish by imposing any axes that the function requires.
* The context stack at this point has these contents
* 	  default context (var=uvar, region=(1:1,1:1,1:1,1:1) for base var)
* big_cx: the merged grid context
*   -->	  argument N context
*	  argument N-1 context
*	  ...
*	  argument 1 context
* So big_cx+1 is passed as a pointer to the argument contexts  
	IF (its_gc) THEN
	   IF (item.EQ.1) THEN
	      CALL GCF_IMPOSE_AXES( big_cx+1, big_cx, uvar, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ENDIF
	
* release context stack space for component just merged
* 1/97: release it if and only if it is not a grid-changing variable.
* For grid changing variable we may need the list of contexts for all of the
* arguments to the GC function in order to determine the imposed axes.
* The depth of the stack needed for this should be 2+nargs
* Note that nargs is always nitem-1 since grid-changing variabless always
* compile to uvar_item_type = arg1 arg2 ... argn gc_function
* BUT that this routine examines items in reverse order (why?) so item=1
* is the key that we are evaluating the last argument for the GC variable.
* Note that in principle there should be no problem handling the stack in
* this manner (popping it only after all arguments are evaluated)
* for all uvars; handling GC variables differently was a choice in
* order to minimize changes to already working code
	IF ( is_phase(isp).NE.0 ) THEN
	   IF ( its_gc ) THEN
	      IF (item.EQ.1) THEN
	         cx_stack_ptr = cx_stack_ptr - (nitem-1)
	      ELSE
	         CONTINUE ! let the cx stack accumulate the argument contexts
	      ENDIF
	   ELSE
	      cx_stack_ptr = cx_stack_ptr - 1
	   ENDIF
	ENDIF
	GOTO 100

* grid for this component is not known.  Back to IS stack to get it
* ... repeat this phase again when we get back
 900	is_phase(isp) = is_phase(isp) - 1
* load the new variable info onto the IS stack
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx   ( isp ) = hi_cx
	is_uvar ( isp ) = cx_variable( hi_cx )
	is_act  ( isp ) = isact_get_grid	! temp diagnostic
	RETURN 1
	
* DONE - clean up
 1000	CONTINUE
* have any variables been encountered ?
* is this a completely abstract expression like X+Y*2 ?
* ... 8/95 check uvar_dset: LET/D variable needs uvar_need_dset always true 
	IF ( is_sum(isp) .EQ. isalg_transfer ) THEN
	   IF ( uvar_dset(uvar).EQ.unspecified_int4 ) THEN
	      uvar_need_dset(uvar) = .FALSE.
	      uvar_grid(uvar, pdset_irrelevant) = mgrid_abstract
	   ELSE		! defined with LET/D
!	      uvar_need_dset(uvar) = .TRUE.	! always TRUE anyway
	      uvar_grid(uvar, dset) = mgrid_abstract
	   ENDIF
	   gnum = mgrid_abstract
	   GOTO 2000
	ELSEIF (uvar_grid(uvar,pdset_irrelevant).EQ.mgrid_abstract) THEN
* ... abstract grid after second pass
	   GOTO 2000
	ENDIF

* catalog this result checking for matching previously existing grid
	CALL TM_GET_LIKE_DYN_GRID(grid, its_dynamic, gnum, 
     .				grd_stk_ptr, status )  
	IF (status .NE. merr_ok) GOTO 5000
	IF ( its_dynamic ) THEN
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('allocate', isact_class_impl_grid, gnum, 0)
	ENDIF

* point uvar to its grid (possibly need different grid for each dset)
* ... 8/95 check uvar_dset: LET/D variable needs uvar_need_dset always true 
	IF ( uvar_dset(uvar).EQ.unspecified_int4 ) 
     .	   uvar_need_dset(uvar) = .NOT.cx_dset_gvn(big_cx)
	IF ( uvar_need_dset(uvar) ) THEN
	   uvar_grid( uvar, dset ) = gnum
	ELSE
	   uvar_grid( uvar, pdset_irrelevant ) = gnum
	ENDIF

* * * SECOND PASS CHECK * * *
* need a second pass to include pseudo_vars in calculation ?
 2000	DO 1150 idim = 1, 4
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_unknown ) THEN
* ... yes. a 2nd pass - re-initialize some things and go do it all again
* (uvar_grid and data set info are NOT re-initialized)
	      DO 1110 i = 1, 4
 1110	      uvar_given(i,uvar) = uvlim_irrelevant
	      is_phase( isp ) = 0
	      is_sum( isp ) = isalg_transfer
	      cx_grid( lo_cx ) = gnum	! set default for pseudo-vars
	      IF ( mode_diagnostic )
     .			CALL DIAGNOSTIC_OUT( 'pass #2', lo_cx, point_to_cx )
	      GOTO 100
	   ENDIF
 1150	CONTINUE

* release context (lo_cx, big_cx), IS and grid stack space used in grid-getting
	IF ( big_cx .NE. cx_stack_ptr
     .	.OR. big_cx .NE. lo_cx+1 ) WRITE (6, *) 'uvar_stk_err' ! temp diag
	cx_stack_ptr = cx_stack_ptr - 2
	isp = isp - 1
	CALL DEALLO_GRID( status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	RETURN 2

* error exit
 5000	CALL DEALLO_ALL_GRIDS
	RETURN
 9000	CALL ERRMSG( ferr_internal, status, 'is_uvar_grid', *5000 )
 9100   IF ( dset .LE. 0 ) GOTO 9000
        CALL GET_SHORT_DSET_NAME( dset, name, i )
        IF ( .NOT. (dp_phil_sieg(dset).OR.dp_cox(dset)) ) THEN
           CALL WARN( name(:i)//' is not a GFDL model data set' ) 
        ELSE
           GOTO 9000
        ENDIF
 	CALL ERRMSG( ferr_invalid_command, status,
     .               'no grid for GFDL diagnostic variable '
     .               //VAR_CODE(cat_calc_var, cx_variable(hi_cx)),
     .               *5000 )
	END	
