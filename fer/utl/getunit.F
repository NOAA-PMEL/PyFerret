	SUBROUTINE GETUNIT (lun, status)

* Find an available unit number.

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/28/90 - Extracted from Don Denbo's PPLUS code, 4 May 90, N. Soreide.
*		   modified for FERRET by Steve Hankin

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER	lun, status

* internal variable declarations:
	LOGICAL lopen

* initialize
	lun=10

*  if the unit is not opened, then it is available.
200	INQUIRE(UNIT=lun,OPENED=lopen)
	IF (lopen) THEN
	   lun=lun+1
	   IF (lun.gt.60) CALL ERRMSG( ferr_internal, status,
     .		     			'too many IO channels', *5000)
	   GOTO 200
	ENDIF

* success
	status = ferr_OK
	RETURN

* error exit
 5000	RETURN
	END
