	SUBROUTINE GET_UVAR_GRID( uvar, dset, status )

* Determine the defining grid for user-defined variable, uvar
* The procedure is performed on the (IS) interpretation stack using stack
* frames on the IS stack, the CX (context) stack and the grid stack.
* In order the determine the requested grid the grids for all of the user
* variables on which the definition depends must also be known or determined.
* Thus, the process is inherently recursive - requiring the stack usage from
* FORTRAN.
* Each stack frame is 1 IS level, 2 CX levels and 1 grid.
* On successful exit from this routine all stacks should be left in their
* original states.

* Example:
*	yes? DEFINE NEW1 = TEMP[Z=@AVE]
*	yes? DEFINE NEW2 = NEW1 * X
*	yes? LIST/Z=0:100/X=180:160W NEW2

* ... to get the grid for NEW2 the grid for NEW1 must be determined.

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/5/89  

* calling argument declarations:
	INTEGER		uvar, dset, status

* internal variable declarations:
	INTEGER		isp_orig, cx_orig, cx, idim

#ifdef unix

! debug includes 11/6/94 *sh*
        include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
        external xgt_grid_data

	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* pre-load the variable info onto the CX stack
	cx_orig = cx_stack_ptr
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	cx = cx_stack_ptr
	cx_data_set(cx) = dset
	cx_variable(cx) = uvar
	cx_category(cx) = cat_user_var

* pre-load the variable info onto the IS stack
	isp_orig = isp
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_cx   ( isp ) = cx
	is_uvar ( isp ) = uvar
	is_act  ( isp ) = isact_get_grid	! temp diagnostic

* initialize the stacks - (repeated for each component)
 100	CALL IS_GRID_MODE( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* evaluate the grid by merging component grids
*  ... exit 100 is to evaluate the grid for a component uvar
*  ... exit 300 means that a grid has been obtained
*  ... "normal" return indicates an error has occurred
 200	CALL IS_UVAR_GRID( *100, *300, status )
	GOTO 5000

* a grid has been obtained.  Are we done ?
 300	IF ( isp .EQ. isp_orig ) THEN
* ... yes - clean up and go home
	   IF ( cx_stack_ptr .NE. cx_orig ) WRITE (6,*) 'usr_v_gd'	! temp
	   RETURN
	ELSE
* ... no - go back for more
	   GOTO 200
	ENDIF

* error exit
 5000	IF ( status .EQ. ferr_stack_ovfl )
     .		CALL WARN( 'Check for recursion in variable definitions' )
	RETURN
	END
