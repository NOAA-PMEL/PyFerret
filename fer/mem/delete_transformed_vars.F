	SUBROUTINE DELETE_TRANSFORMED_VARS( status )

* delete all regridded or transformed variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/10/89
* V200:  6/9/89 - deleted variables flagged in mr_protected

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	INTEGER	mr, idim

* do it (entry max_mrs is the dummy slot)
	DO 100 mr = 1, max_mr_avail
	   IF ( mr_protected(mr) .EQ. mr_deleted ) GOTO 100

* transforms ?
	   DO 50 idim = 1,4
  50	   IF ( mr_trans(idim,mr) .NE. trans_no_transform ) GOTO 60
* regridded ?
	   IF ( .NOT.mr_unstand_grid(mr) ) GOTO 100

* delete it if it's not permanent
  60	   IF ( mr_protected( mr ) .EQ. mr_perm_protected ) THEN
	      CALL ERRMSG( ferr_perm_var, status,
     .			'use SHOW MEMORY/PERM', *5000 )
	   ELSE
	      CALL DELETE_VARIABLE( mr )
	   ENDIF

 100	CONTINUE
	status = ferr_ok

 5000	RETURN
	END
