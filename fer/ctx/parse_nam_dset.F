	SUBROUTINE PARSE_NAM_DSET( full_var_name, default_cx,
     .				       dset, cat, var, mods_cx, status )

* given a full variable specification of the form:
*			(eg.) TEMP[D=n,G=gname] (with optional region info)
* return context information describing variable and data set

* Note that "dset" is both an input and output:  on input it is the
*		default data set to be searched.  On output it is the
*		data set that is correct.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V402: 7/95 - extracted from PARSE_NAM_DSET_GRD (w/out the grid part)
* V450: 1/97 - use local variables instead of xPROG_STATE vars for item_start
*		etc. to avoid conflict. (But share the max_item_list parameter
*		from the COMMON area.
	
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, dset, cat, var, mods_cx, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	INTEGER		IGRID, var_cx, stk_stat,
     .			full_name_len, left_bracket, right_bracket,
     .			name_len, brac_nest, i
 	LOGICAL		name_only
* 1/97 addition
	INTEGER	num_it, it_start(max_item_list), it_end(max_item_list)


* set up error flag in case we fail
	var = munknown_var_name

* variable context level
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	var_cx = cx_stack_ptr
	CALL TRANSFER_CONTEXT ( default_cx, var_cx )

* modifications context level
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 4950
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* separate simple name from possible modifiers
	full_name_len = LEN   ( full_var_name )
	left_bracket  = INDEX ( full_var_name, '[' )
	name_only     = left_bracket .EQ. 0
	IF ( name_only ) THEN
	   name_len = full_name_len
	ELSE
	   name_len = left_bracket - 1
* ... hunt for matching right bracket
	   brac_nest = 1
	   right_bracket = 0	! default to catch syntax errors
	   DO 200 i = left_bracket+1,full_name_len
	      IF ( full_var_name(i:i) .EQ. "[" ) THEN
	         brac_nest = brac_nest + 1
	      ELSEIF ( full_var_name(i:i) .EQ. "]" ) THEN
	         brac_nest = brac_nest - 1
	         IF ( brac_nest .EQ. 0 ) THEN
	            right_bracket = i
	            GOTO 210
	         ENDIF
	      ENDIF
 200	   CONTINUE
 210	   IF (   right_bracket .LE. left_bracket + 1
     .	     .OR. name_len      .LE. 0			) GOTO 5100
	ENDIF

* break up the "[...,...,...]" into comma-separated arguments
	IF ( .NOT.name_only ) THEN
	   CALL PARSE_COMMA_LIST (	full_var_name,
     .					left_bracket  + 1,
     .					right_bracket - 1,
     .					max_item_list,
     .					num_it,
     .					it_start,
     .					it_end,
     .					status		)
	   IF ( status .NE. ferr_ok ) GOTO 4900
	ELSE
	   num_it = 0
	ENDIF

*interpret the list of items - unknown items represent errors
	CALL GET_CONTEXT_MODS (	default_cx,
     .				full_var_name,
     .				mods_cx,
     .				max_item_list,
     .				num_it,
     .				it_start,
     .				it_end,
     .				unknown_qual_bad,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 4900

* apply data set to context if specified
	IF ( cx_data_set( mods_cx ) .NE. unspecified_int4 )
     .		cx_data_set( var_cx ) = cx_data_set( mods_cx )
	IF (dset.EQ.unspecified_int4 .OR. cx_dset_gvn(mods_cx))
     .		dset = cx_data_set( var_cx )

* identify the simple variable name
* (var = munknown_var_name will flag failure for the whole routine)
	CALL FIND_VAR_NAME( dset, full_var_name(1:name_len), cat, var )

* relinquish the stack space used for the variable
 4900	CALL STACK_PTR_DN( cx_stack_ptr, cx_stack_ptr_base, stk_stat )

* relinquish the stack space used for the modifications
 4950	CALL STACK_PTR_DN( cx_stack_ptr, cx_stack_ptr_base, stk_stat )
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, full_var_name, *4900 )
 5200	CALL ERRMSG( ferr_unknown_variable, status, full_var_name, *4900 )
! 5300	CALL ERRMSG
!     .		( ferr_unknown_data_set, status,'for '//full_var_name, *4900 )
	END
