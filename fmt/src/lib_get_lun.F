	SUBROUTINE LIB_GET_LUN (lun)

* Find an available unit number.
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/28/90 - Extracted from Don Denbo's PPLUS code, 4 May 90, N. Soreide.
*		   modified for FERRET by Steve Hankin
* Ultrix port 1/17/91: renamed from GETUNIT to LIB_GET_LUN
*		     - replaced status return with STOP
*		     - replaced GOTO loop with DO loop

* calling argument declarations:
	INTEGER	lun

* internal variable declarations:
	LOGICAL lopen

*  if the unit is not opened, then it is available
*	DO 100 lun = 10, 60         ! PLOT+ conflict ???
	DO 100 lun = 61,120
	   INQUIRE(UNIT=lun,OPENED=lopen)
	   IF (.NOT.lopen) RETURN
 100	CONTINUE
	STOP 'insufficient IO channels: LIB_GET_LUN'

	END
