	SUBROUTINE IS_UVAR_GRID( *, *, status )

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
*       *sh*  2/99 - if ABSTRACT grid or normal axis then use XABSTRACT (or
*                       YABSTRACT, ZABSTRACT, TABSTRACT)
*		     A significant consequence of this is that 2 passes are
*		     needed ANY TIME that pseudo-variables which do not state
*		     their own axes are used, becuase they may be generating
*		     abstract axes. For example, the grid of "SST+Z"
* v500: *kob* 3/99 - fixed which crashed ferret when  let/d didn't have a 
*                    dataset present
*	*sh* 3/99 - bug fix: bug introduced with string arguments 
*	*sh* 4/99 - bug fix: 2/99 2nd pass check "need_another_pass" state
*		is lost is a component is fetched afterwards. Instead, save
*		this state as a negative is_phase
*       *kob* 4/99 - bug fix: using /d in the definition of a dataset
*                    independent variable caused a crash.
* V510: 8/99 *sh* - allow G=uvar on both uvars and pseudo-vars.
*			If uvar grid is unknown, go get it!
*	1/00 *sh* - bug fix for nested GC variables 
*	4/00 *sh* - netCDF grids are now dynamic so 2nd pass check to deallo
*			dynamic grid needs logic change
* V530 10/00 *sh* - fixed bug that was revealed by new ability of
*		    GCF_IMPOSE_AXES to take an error exit
*	          - process string variables so that they may be obtained
*		    by EFs during custom axis creation
* V533 7/01 *sh* - save the data type of the result
*	         - process alg_if_marker in support of IF cond THEN string
* V540 10/01 *sh* - fix bug in IF-THEN hack when 2nd pass is required
* V541  4/02 *sh* - fix bug in IF-THEN: err540_data_type.jnl
*                   constants were not processed, so PLOT IF X GT 5 THEN 1
*                   failed to pick up the data type of the constant 1
* v552: 4/03 *acm*  Increase length of variable names to 128 characters
*                   (variables are up to 128, so **name** needs extra space
*                   i.e. for a transformation specifier)
* v552: 5/03 *acm*  Use just the current item when dealing w/ G=uvar with 
*                   target grid not yet known (intermediate variable buff)
* v554: 1/04 *acm* Counter variables for REPEAT/RANGE=/NAME=
* V600 8/05 *acm* - Add variable type attrib_val for definitions of the form
*                   varname.attname

* calling argument declarations:
	INTEGER	status
* normal return is for error conditions
* exit #1 is to request grid for a component ( stack has been pushed )
* exit #2 is to signal that the grid has been obtained and stored in uvar_grid

* internal variable declarations:
	LOGICAL	ITSA_GCVAR, its_dynamic, its_gc, need_another_pass,
     .		saved_ifthen_type, do_err
	INTEGER KNOWN_GRID, STR_SAME,
     .		TM_LENSTR1, GCF_PARENT_UVAR, GCF_PARENT_CHAR, 
     .		 hi_cx, lo_cx, big_cx, start, end, grid,
     .		i, dset, uvar, item, itype, idim, gnum, nitem, uv_grd,
     .		var, len_name, pass_number, cx_save, this_dset,
     .		ifthen_data_type, varid, alen, attype, attlen,
     .          attoutflag

	CHARACTER VAR_CODE*4, name*512, buff*2048, varname*512, 
     .            attname*128

        include 'netcdf.decl'
#include "netcdf.inc"
	include	'tmap_errors.parm'
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xfr_grid.cmn'
	include	'xdset_parms.cmn'

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level to get grids for components
*	 isp	<--	current level: evaluate user-variable grid

* --- end of introductory code ---

* initialize
	lo_cx  = is_cx( isp )
	big_cx = is_big_cx( isp )
	grid   = cx_grid(big_cx)
	uvar   = is_uvar( isp )
	dset   = cx_data_set( lo_cx )
	nitem  = uvar_num_items( uvar )
	its_gc = ITSA_GCVAR(uvar)
* ... initialization to support 2/99 pass-2 changes
	pass_number = 1
! 4/99
	need_another_pass = is_phase(isp) .LT. 0  ! saved from last time here
	is_phase(isp) = ABS( is_phase(isp) )
	cx_save = cx_stack_ptr		! restored on pass #2 (2/99)
	saved_ifthen_type = .FALSE.

* process the next item in the expression from the uvar buffer (last to first)
 100	is_phase(isp) = is_phase(isp) + 1
	item = nitem - is_phase(isp)	+ 1	! point into uvar_text
	IF ( item .LT. 1 ) GOTO 1000		! have processed entire def'n
	itype = uvar_item_type ( item, uvar )
	start = uvar_item_start( item, uvar )
	end   = uvar_item_end  ( item, uvar )
	IF     ( itype .EQ. alg_pseudo_var  ) THEN
	   GOTO 200
	ELSEIF ( itype .EQ. alg_const_var   ) THEN
	   GOTO 250
	ELSEIF ( itype .EQ. alg_child_var   ) THEN
	   GOTO 300
	ELSEIF ( itype .EQ. alg_variable    ) THEN
	   GOTO 305
	ELSEIF ( itype .EQ. alg_attrib_val  ) THEN
	   GOTO 315
	ELSEIF ( itype .EQ. alg_string      ) THEN
	   GOTO 400
	ELSEIF ( itype .EQ. alg_constant    ) THEN
	   GOTO 410
	ELSEIF ( itype .EQ. alg_counter_var   ) THEN
	   GOTO 420
	ELSEIF ( itype .EQ. alg_then_marker ) THEN  ! marks where the IF was
	   GOTO 450
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
	IF ( cx_grid(hi_cx) .LT. unspecified_int4 ) THEN
	   GOTO 800
	ELSEIF ( cx_grid(hi_cx) .NE. unspecified_int4 ) THEN
	   GOTO 500		! yup - merge it
	ELSE
* nope! Set flag indicating a 2nd pass is needed
	   uvar_given( idim, uvar ) = uvlim_unknown  ! ?? irrelevant from 2/99
	   need_another_pass = .TRUE.
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
 300	CALL GCF_CHILD_VAR_NAME( GCF_PARENT_UVAR(uvar),
     .				 GCF_PARENT_CHAR(uvar,start), name )
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
* handle variable regridded to a uvar which has unknown grid (8/99) 
	IF ( cx_unstand_grid(hi_cx)
     . .AND. cx_grid(hi_cx) .LT. unspecified_int4 ) GOTO 800
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

* variable.attname: parse name, get needed parts of context for attribute

 315	name = uvar_text(uvar)(start:end)
	len_name = end-start+1


 	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	hi_cx = cx_stack_ptr
	CALL GET_VAR_CONTEXT (  name(:len_name),
     .				lo_cx,
     .				hi_cx,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

        dset = cx_data_set(hi_cx)
        IF (dset .EQ. pdset_irrelevant) dset = -1
        do_err = .TRUE.
        attname = ' '
        name = ' '
        CALL  BREAK_VARATTNAME (  uvar_text(uvar)(start:end), dset, 
     .                            varname, attname, varid, do_err, status )

* Get attribute type and length
        
* See if it is a pseudo-attribute attnames, ndims, etc.

        attype = NCFLOAT

        IF (STR_SAME(attname(:4), 'attn') .EQ. 0  .OR. 
     .       STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN
           attype = NCCHAR
        ELSE IF (STR_SAME(attname(:4), 'ndim') .EQ. 0  .OR.
     .      STR_SAME(attname(:4), 'type') .EQ. 0  .OR.
     .      STR_SAME(attname(:4), 'nvar') .EQ. 0  .OR.
     .      STR_SAME(attname(:4), 'natt') .EQ. 0 ) THEN
           attype = NCFLOAT

        ELSE IF (STR_SAME (varname, '.') .EQ. 0) THEN
           IF (STR_SAME(attname(:4), 'varn') .EQ. 0   .OR.
     .         STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN
              attype = NCCHAR

           ELSE IF (STR_SAME(attname(:3), 'att') .EQ. 0   .OR.
     .           STR_SAME(attname(:4), 'ndim') .EQ. 0  .OR.
     .           STR_SAME(attname(:3), 'nat') .EQ. 0  .OR.
     .           STR_SAME(attname(:4), 'nvar') .EQ. 0 ) THEN
              attype = NCFLOAT
           ENDIF

* Otherwise get att type from the linked-list structure

        ELSE
           CALL CD_GET_VAR_ATT_INFO (dset, varid, attname, attype,
     .           attlen, attoutflag, status)
        ENDIF

        IF (attype .EQ. NCCHAR) THEN
           cx_type(hi_cx) = ptype_string
	ELSE
	   cx_type(hi_cx) = ptype_float
	ENDIF
        
* set the X axis limits equal to the length of the array
c	cx_lo_ss(hi_cx, x_dim) = 1
c	cx_hi_ss(hi_cx, x_dim) = attlen !!
        IF (attype .EQ. NCCHAR) cx_hi_ss(hi_cx, x_dim) = 1

* The grid is abstract 

	var = cx_variable(hi_cx)
        uv_grd = KNOWN_GRID( cx_data_set(hi_cx), cat_attrib_val, var )

	IF ( cx_unstand_grid(hi_cx)
     . .AND. cx_grid(hi_cx) .LT. unspecified_int4 ) GOTO 800


* do we know the defining grid for the component ?
* (not necessarily the same as the returned grid --> there may be a regrid )

	IF ( cx_grid(hi_cx) .NE. unspecified_int4 ) THEN
	   GOTO 500	! know grid --> merge it
	ELSE
	   GOTO 9000	! err: grid for varname.attname is known
	ENDIF

        GOTO 500

* string constant
 400	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	hi_cx = cx_stack_ptr
        CALL STRING_CONTEXT (   uvar,
     .                          item,
     .                          lo_cx,
     .                          hi_cx,
     .                          status )
        IF ( status .NE. ferr_ok ) RETURN
	GOTO 500		! go merge it

* constant (float)
* this block is only for the purpose of setting the data type of the result
* ... in case the result is only a constant (in an IF-THEN expression)
! ... The grid is unspecified for a constant
 410	IF ( is_sum(isp) .EQ. isalg_transfer ) THEN
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   hi_cx = cx_stack_ptr
           CALL CONSTANT_CONTEXT ( uvar_text(uvar)(start:end),
     .                             lo_cx,
     .                             hi_cx,
     .                             status )
           IF ( status .NE. ferr_ok ) RETURN
	   cx_type(big_cx) = cx_type(hi_cx)
	   cx_stack_ptr = cx_stack_ptr - 1
	ENDIF
	GOTO 100

* counter variable (repeat/range=)
* this block is only for the purpose of setting the data type of the result
! ... The grid is unspecified for a constant
 420	IF ( is_sum(isp) .EQ. isalg_transfer ) THEN
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   hi_cx = cx_stack_ptr
           name = '1'
           CALL CONSTANT_CONTEXT ( name,
     .                             lo_cx,
     .                             hi_cx,
     .                             status )
           IF ( status .NE. ferr_ok ) RETURN
	   cx_type(big_cx) = cx_type(hi_cx)
	   cx_stack_ptr = cx_stack_ptr - 1
	ENDIF
	GOTO 100

* "IF" marker in an "IF condition THEN result1 ELSE result2" structure
* ... The data type of the result depends only on the result clauses
* ... prepare to restore the data type after the "condition" is processed
* ... Note: expressions are processed here from high item backwards so
*           the "condition" is the last thing to be procesed 
 450	saved_ifthen_type = .TRUE.
	ifthen_data_type = cx_type(big_cx)
	GOTO 100

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
	IF (saved_ifthen_type) cx_type(big_cx) = ifthen_data_type

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
	   IF (.NOT.its_gc) THEN
	      cx_stack_ptr = cx_stack_ptr - 1
	   ENDIF
	ENDIF
	GOTO 100

* deal with G=uvar with target grid not yet known -- get the target grid (8/99)
 800	CONTINUE
* ... decode the character position of the "G=" target variable name
* 5/03 bug fix acm: use buff here for just the current item, not the 
*                   whole uvar_text(uvar) 
* 
        buff = uvar_text(uvar)(start:end)
	end = (-cx_grid(hi_cx))/2000
	start = (-cx_grid(hi_cx)) - 2000*end + unspecified_int4
* 2/03 *kob* g77 port - need the '*' on call to errmsg to make it an
*                       alternate return label, rather than passing
*                       by reference
	IF ( start .LT. 0
     .  .OR. end   .LT. start
     .	.OR. end   .GT. cmnd_buff_len ) CALL ERRMSG(
     .			ferr_internal, status, 'G=uvar', *5000)

	name = buff(start:end)
	len_name = end - start + 1
* ... the context stack is already pushed from above
	CALL GET_VAR_CONTEXT (  name(:len_name),
     .				lo_cx,
     .				hi_cx,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000


* grid for this component is not known.  Back to IS stack to get it
* ... repeat this phase again when we get back
 900	is_phase(isp) = is_phase(isp) - 1
	IF (need_another_pass) is_phase(isp) = -1 * is_phase(isp)  ! 4/99
* load the new variable info onto the IS stack
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) RETURN
	is_cx   ( isp ) = hi_cx
	is_uvar ( isp ) = cx_variable( hi_cx )
	is_act  ( isp ) = isact_get_grid	! temp diagnostic
	RETURN 1
	
***  DONE - clean up
 1000	CONTINUE
* for grid-changing vars all of the arguments contextx are still on the stack 
	IF ( its_gc ) THEN
	   DO 1010 i = 1, nitem - 1	! 3/99 - strings not on stack
	      cx_stack_ptr = cx_stack_ptr - 1
!	      IF (uvar_item_type(i,uvar).NE. alg_string)
!     .			cx_stack_ptr = cx_stack_ptr - 1
 1010	   CONTINUE
	ENDIF
* have any variables been encountered ?
* is this a completely abstract expression like X+Y*2 ?
* ... 8/95 check uvar_dset: LET/D variable needs uvar_need_dset always true 
	IF ( is_sum(isp) .EQ. isalg_transfer ) THEN
	   IF ( uvar_dset(uvar).EQ.unspecified_int4 ) THEN
	      uvar_need_dset(uvar) = .FALSE.
	      uvar_grid(uvar, pdset_irrelevant) = mgrid_abstract
	      uvar_data_type(uvar, pdset_irrelevant) = ptype_float
	   ELSE		! defined with LET/D
!	      uvar_need_dset(uvar) = .TRUE.	! always TRUE anyway
* *kob* 4/99 bug fix - check to see if let/d is dataset independent
	      IF (dset .EQ. unspecified_int4) THEN
		 uvar_grid(uvar, pdset_irrelevant) = mgrid_abstract
	         uvar_data_type(uvar, pdset_irrelevant) = ptype_float
              ELSE
                 uvar_grid(uvar, dset) = mgrid_abstract
	         uvar_data_type(uvar, dset) = ptype_float
	      ENDIF
	   ENDIF
	   gnum = mgrid_abstract
	   CALL TM_USE_DYN_GRID( mgrid_abstract )
	   GOTO 2000
	ELSEIF (uvar_grid(uvar,pdset_irrelevant).EQ.mgrid_abstract) THEN
* ... abstract grid after second pass
* (2/99 comment this out since pseudo-vars now determine their own grids:
*	XABSTRACT, YABSTRACT, ZABSTRACT, or TABSTRACT
!	   GOTO 2000	! removed 2/99
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
* ... 3/99 but LET/D ("default") needs uvar_need_dset FALSE 
* ... 11/02 LET/D needs uvar_need_dset FALSE only when there is no dataset
*           associated with the variable.  In the case where LET/D wants the
*           default dataset associated with the variable, it should be true
*           so remove the 3/99 additions - which fixed a bug which occured
*           only when a let/d was done and no dataset was initialized.  This
*           was the incorrect place to fix that bug.

        IF ( uvar_dset(uvar).EQ.unspecified_int4 )
     .            uvar_need_dset(uvar) = .NOT.cx_dset_gvn(big_cx)

c        IF ( itype .EQ. alg_attrib_val  ) THEN
c           uvar_dset(uvar) = dset
c           uvar_need_dset(uvar) = .TRUE.
c        ENDIF

	IF ( uvar_need_dset(uvar) .AND. dset .NE. unspecified_int4) THEN
	   this_dset = dset
	ELSE
	   this_dset = pdset_irrelevant
	ENDIF

	uvar_grid( uvar, this_dset ) = gnum
	uvar_data_type(uvar, this_dset) = cx_type(big_cx)

* * * SECOND PASS CHECK * * *
* need a second pass to include pseudo_vars in calculation ?
 2000	IF ( need_another_pass ) THEN 
* ... uvar_grid and data set info are NOT re-initialized
	   IF ( pass_number .GT. 1 ) GOTO 9000	! caught in a loop ...
	   DO 1110 i = 1, 4
 1110	   uvar_given(i,uvar) = uvlim_irrelevant
	   uvar_grid( uvar, this_dset ) = unspecified_int4 ! 10/00
	   uvar_data_type(uvar, this_dset) =  unspecified_int4 ! housekeeping
	   is_phase( isp ) = 0
	   is_sum( isp ) = isalg_transfer
	   cx_stack_ptr = cx_save
	   need_another_pass = .FALSE.
	   pass_number = pass_number + 1
	   saved_ifthen_type = .FALSE.
	   cx_grid( lo_cx ) = gnum	! set default for pseudo-vars
* ... deallocate the grid just obtained since it will be
*     allocated again on the next pass
! pre-4/00 IF (gnum .GT. max_grids ) CALL TM_DEALLO_DYN_GRID( gnum )
	   CALL TM_DEALLO_DYN_GRID( gnum )
* ... release the use-count hold that the temporary grid has on the axes
*	since it will be re-asserted on the second pass
*	We need to retain the grid, itself, because it is used on pass 2
	   DO 1120 i = 1, 4
	      CALL TM_DEALLO_DYN_LINE(grid_line(i,grid))
 1120	   CONTINUE
	   IF ( mode_diagnostic ) THEN
	      CALL DIAG_OP('dealloc ',isact_class_impl_grid,gnum,0)
	      CALL DIAGNOSTIC_OUT( 'pass #2', lo_cx, point_to_cx )
	   ENDIF
	   GOTO 100
	ENDIF

* release context (lo_cx, big_cx), IS and grid stack space used in grid-getting
	IF ( big_cx .NE. cx_stack_ptr
     .	.OR. big_cx .NE. lo_cx+1 ) WRITE (6, *) 'uvar_stk_err' ! temp diag
	cx_stack_ptr = cx_stack_ptr - 2
	isp = isp - 1
	CALL DEALLO_GRID( status )
	IF ( status .NE. ferr_ok ) GOTO 5000
c        IF ( itype .EQ. alg_attrib_val  ) THEN
c           uvar_dset(uvar) = unspecified_int4  ! or irrelevant??
c        ENDIF

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
