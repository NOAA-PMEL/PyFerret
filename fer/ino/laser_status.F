
	SUBROUTINE LASER_STATUS( string, status )

* check the status returned by the last command sent to the Panasonic OMDR

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvideo.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVIDEO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	string

* local variable declarations:
	INTEGER		TM_LENSTR1, i

* diagnostic output
	IF ( mode_diagnostic )
     .		WRITE (ttout_lun, 3000 ) odr_resp(:TM_LENSTR1(odr_resp))
 3000	FORMAT (' ODR: ',A)

* set return status
	IF ( odr_resp(1:1) .EQ. 'E' ) THEN
	   status = ferr_ODR_error
	ELSE
	   status = ferr_ok
	ENDIF

* always pass back error string
	string = odr_resp

	RETURN
	END

