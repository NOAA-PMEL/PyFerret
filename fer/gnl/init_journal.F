	SUBROUTINE INIT_JOURNAL( status )

* open the FERRET journal file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  5/93 - extracted from initialize.F
* V312:  5/94 - insert "PROCLAIM" output from here instead of from INITIALIZE
* V320: 12/29/94 - use WARN to avoid direct tty output (for GUI)

#ifdef unix
	include 'tmap_format/tmap_errors.parm'  ! error codes
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'  ! error codes
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE '($SSDEF)'				! LIB_ return status
	INCLUDE '($JPIDEF)'				! for VMS mode code
#endif

* calling argument declarations:
	INTEGER status

* local variable declarations:
	INTEGER TM_LENSTR1, LIB_GET_LUN, jnl_len

* get logical unit to open the journal file
	status = LIB_GET_LUN( jrnl_lun )

*  rename last journal file by this name to old version
#ifdef unix
        CALL TM_PARK_LAST_VERSION( journal_file, status )
        IF ( status .NE. merr_ok ) GOTO 9000
	OPEN (	FILE=journal_file,
     .		UNIT=jrnl_lun,
     .		STATUS='NEW',
     .		ERR = 9000 )
#else
* VMS ... make sure "FERRET_JOURNAL" has been defined
!	status = LIB$SYS_TRNLOG( jrnl_file, ,jrnl_trans )
!	IF ( status .EQ. ss$_notran ) STOP 'FERRET_JOURNAL must be defined'
	OPEN (	FILE=journal_file,
     .		UNIT=jrnl_lun,
     .		CARRIAGECONTROL = 'LIST',
     .		RECL = 255,
     .		STATUS='NEW',
     .		ERR = 9000 )
#endif

* put introductory remarks into journal file
	CALL PROCLAIM ( jrnl_lun, '! ' )

* successful completion
	status = ferr_ok   ! status not really used
	RETURN

* errors
* note: error not reported to ERRMSG since this is called from INITIALIZE
 9000	jrnl_lun = unspecified_int4	! flag that there is no journal file
	jnl_len = TM_LENSTR1(journal_file)
	CALL WARN('Unable to create journal file '
     .			//journal_file(:jnl_len) )
	status = ferr_invalid_command  

	END

