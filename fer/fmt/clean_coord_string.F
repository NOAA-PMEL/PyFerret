	SUBROUTINE CLEAN_COORD_STRING ( idim, string )

* replace the blank in "dd-mmm-yyyy hh:mm:ss" with a colon
* axe the time field from climatological dates: "dd-mmm      hh:mm" -> "dd-mmm"
* replace 19E(379) with the unambiguous 379E

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V300:  8/27/93

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER idim
	CHARACTER*(*) string

* internal variable declarations:
	INTEGER		TM_LENSTR1, i, i0, slen

	IF ( idim.EQ.y_dim .OR. idim.EQ.z_dim ) RETURN

	slen = TM_LENSTR1( string )

	IF ( idim .EQ. t_dim ) THEN
* FIX TIMES
	   IF ( slen .LE. 11 ) RETURN   	! date, only - no time
	   IF ( string(3:3) .NE. '-' ) RETURN	! not a date string

	   IF ( string(9:9) .EQ. ' ' ) THEN
* wipe out time on climatological dates
	      DO 100 i = 7, slen
	        string(i:i) = ' '
 100	      CONTINUE
	   ELSE
	     string(12:12) = ':'
	   ENDIF

	ELSE
* FIX LONGITUDES
	   i0 = INDEX( string, '(' )      ! e.g."19E(379)"
	   IF ( i0 .GT. 0 ) THEN
	      DO 200 i = i0+1, slen-1     ! --> "379(379)"
 200	      string(i-i0:i-i0) = string(i:i)
	      DO 210 i = slen-i0, slen    ! --> "379     "
 210	      string(i:i) = ' '
	      slen = slen-i0
	      string(slen:slen) = 'E'     ! --> "379E"
	   ENDIF

	ENDIF

	RETURN
	END
