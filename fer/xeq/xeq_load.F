	SUBROUTINE XEQ_LOAD( memory )

* execute the LOAD command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/4/87
* revision 0.1 - 9/16/87 - added want_cmnd_data to GET_CMND_DATA call
* revision 2.0 - 3/10/88 - added LOAD/NAME=
* revision 2.1 - 3/23/88 - enlarged cx_DIM_STR to accomodate date/time strings
* revision 2.2 - 8/15/88 - changed /NAME= re-cataloging procedure for new
*			   memory management routines
* revision 2.3 - 10/7/88 - remove in-plane transforms, too and re-catalog
* revision 2.4 -11/21/88 - new TM IO library GET_LIKE_GRID --> TM_GET_LIKE_GRID
*					     TRANSFER_GRID --> TM_COPY_GRID
* V200:  6/9/89 - deleted variables flagged in mr_protected
*		- /NAME= eliminated
*	12/15/89- fixed error check for LOAD/PERM expression
*	  5/1/90- added code so LOAD/TEMP works
*		  still flawed: doesn't detect region gt requested
*		- and bixed del_flink/blink bug in LOAD/PERM
* V230: 8/11/92 - double load of /PERMANENT got subscript error
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* local parameter definitions:
	INTEGER	slash_perm, slash_temp, slash_name
	PARAMETER     ( slash_temp = 1,		! (default)
     .			slash_perm = 2,
     .			slash_name = 12 )

* calling argument declarations:
	REAL	memory(*)

* local variable declarations:
	LOGICAL		EXPLICIT_UVAR, permanent, named, temp
	INTEGER		status, iload, mr, next, prev, last_head

* initialize
	named	  = qual_given( slash_name ) .GT. 0
	permanent = qual_given( slash_perm ) .GT. 0
	temp	  = qual_given( slash_temp ) .GT. 0

* save the data under a given name
	IF ( named ) CALL ERRMSG( ferr_invalid_command, status,
     .		'/NAME= no longer supported - use DEFINE VARIABLE', *5000 )
	
* get the requested data somehow
	CALL GET_CMND_DATA ( memory, cx_last, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* set protections as indicated (check for already protected 8/92)
	IF ( permanent ) THEN
	   DO 400 iload = 1, num_uvars_in_cmnd
	      mr = is_mr( iload )
	      IF ( mr_category( mr ) .EQ. cat_user_var ) THEN
	         IF ( .NOT.EXPLICIT_UVAR(mr_variable(mr)) ) CALL ERRMSG
     .			( ferr_invalid_command, status,
     .			'Cannot LOAD/PERM "expression".  Use DEFINE VARIABLE',
     .			 *5000 )
	      ENDIF
              IF ( mr_protected(mr) .NE. mr_perm_protected ) THEN  ! 8/92
	         mr_protected( mr ) = mr_perm_protected
* ... remove from deletion priority chain
	         prev = mr_del_blink(mr)
	         next = mr_del_flink(mr)
	         mr_del_flink(prev) = next
	         mr_del_blink(next) = prev
	         mr_del_flink( mr ) = unspecified_int4	! temp diag
	         mr_del_blink( mr ) = unspecified_int4	! temp diag
              ENDIF
 400	   CONTINUE

* LOAD/TEMPORARY - explicitly specified to reverse /PERMANENT
	ELSEIF ( temp ) THEN
	   DO 500 iload = 1, num_uvars_in_cmnd
	      mr = is_mr( iload )
	      IF ( mr_protected( mr ) .EQ. mr_perm_protected ) THEN
	         mr_protected( mr ) = mr_not_protected
* ... restore at head of deletion priority chain
	         last_head = mr_del_flink(0)
	         mr_del_blink(mr) = 0
	         mr_del_flink(mr) = last_head
	         mr_del_flink(0)  = mr
	         mr_del_blink(last_head) = mr
	      ENDIF
 500	   CONTINUE

	ENDIF

	RETURN

* error exit(s)
 5000	RETURN
	END
