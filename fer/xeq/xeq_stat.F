	SUBROUTINE XEQ_STAT( memory )

* execute the STAT command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/25/88
* V200:  5/19/89 - 4D symmetrical
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
* V312: 5/94 - array "memory" as a calling argument
*       10/26/94 - changed "STAT" to "VAR_STAT" for IBM/AIX
* V420	 2/7/96  - added STAT/BRIEF

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	full
	INTEGER	mv, cx, istat, status
	
* decode the qualifiers
	full = qual_given( slash_stat_brief ) .EQ. 0

* get the requested data somehow
	CALL GET_CMND_DATA ( memory, cx_last, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* compute and display the results
  	DO 400 istat = 1, num_uvars_in_cmnd
	   mv = is_mr( istat )
	   cx = is_cx( istat )
	   CALL NON_ARRAY_SUBSC( mv, 1 )
	   CALL VAR_STAT( memory(1, mr_blk1(mv)), mv, cx,
     .			  ttout_lun, full )
 400	CONTINUE

	RETURN

* error exit(s)
 5000	RETURN
	END
