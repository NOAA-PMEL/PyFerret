c*
c* ef_convert_date_string.F
c*
c* Ansley Manke * 
c* 12/2018
c*
c*
c* This routine converts a date string of the form yyyy-mm-dd hh:mm:ss to
c* the default format dd-MMM-yyyy hh:mm:ss
c*

      SUBROUTINE ef_convert_date_string(datestring, status)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'

* argument declarations
      INTEGER status
      CHARACTER*(*) datestring

* Internal declarations
      INTEGER infmt, outfmt, cal
      CHARACTER*30 buff

* Initialize
      infmt = pdate_numslashnum
      outfmt = pdate_vax
      cal = gregorian  ! doesn't matter just for reformatting a date

      status = 0  ! just local setting not Ferret-status code
      IF (datestring(5:5) .NE. '-' ) RETURN

      CALL TM_DATE_REFORMAT( datestring, infmt, cal, buff, outfmt, .FALSE., status)
      IF (status .EQ. 3) THEN
         status = 0
	 datestring = buff
      ENDIF

      RETURN 
      END
