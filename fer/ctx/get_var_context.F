	SUBROUTINE GET_VAR_CONTEXT
     .		      ( full_var_name, 
     .			default_cx,
     .			var_cx,
     .			status )

* given a full variable specification of the form:
*			(eg.) TEMP[X=170E:170W,K=+1D,T=188979]
* return in var_cx context information describing the exact variable instance
* including variable, data set, grid, region and plane

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/15/86
* revision 0.1 - 3/23/87 - added new TMAP library: time axes, file var. names,
*						   bad data flag
*			 - context mods from var name applied IN PARSE_VAR_NAME
* revision 0.2 - 5/5/88  - FINISH_CONTEXT_MODS --> FLESH_OUT_CONTEXT
*			   added CONFINE_CONTEXT
* V200:  6/22/89 - eliminated GT_CONTEXT_PLANE
*		 - added oversizing of contexts
*		 - added test for irrelevant data set
	
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, var_cx, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	INTEGER		mods_cx, idim

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, var_cx )

* find a context slot for the modifications that may be attached to the var name
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* get variable and attached modifications
	CALL PARSE_VAR_NAME ( full_var_name, var_cx, mods_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* insert bad data flag for variable
	CALL GET_CONTEXT_BAD_FLAG( var_cx )

* determine the defining grid
	CALL GET_CONTEXT_GRID ( var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* can't do more if the grid isn't known
	IF ( cx_grid(var_cx) .EQ. unspecified_int4 ) GOTO 100

* data set info may be irrelevant
* indicate that now so memory searches will be correct
	IF ( cx_category(var_cx) .EQ. cat_user_var ) THEN
	   IF ( .NOT.uvar_need_dset(cx_variable(var_cx)) ) THEN
	      cx_data_set(var_cx) = pdset_irrelevant
*  ... flag that all data set info is contained within the variable
	      cx_dset_gvn(var_cx) = .TRUE.
	   ENDIF
	ENDIF

* confine limits to grid
	CALL CONFINE_CONTEXT ( var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* flesh out the context on each axis with grid-related positions
	DO 10 idim = 1, 4
	   CALL FLESH_OUT_AXIS( idim, var_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
 10	CONTINUE

* enlarge the context ? (contour plots need data extending beyond plot region)
	IF ( cx_ovsiz( var_cx) .NE. ovsiz_off ) CALL OVSIZ_CX( var_cx )

* relinquish the stack space used for the modifications
 100	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	RETURN

* error exit
 5000	RETURN
	END
