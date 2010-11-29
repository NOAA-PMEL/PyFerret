      SUBROUTINE TM_SET_CURRENT_CALENDAR (set_name, status)

C     Set the current calendar name and id in common current_calendar
C     and loads calendar info into common.
C
C     Returns merr_varlim if calendar is invalid, merr_ok successful completion
C     J Davison 10.99
C     V 1.0

C  11/29/00  Ansley Manke  revise to match just 3 characters of allowed_calendars
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

#ifdef unix
      include 'calendar.decl'
      include 'calendar.cmn'
      include 'tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif 
      integer         end, ndx, status
      character*255   test_string

      character*16    cal_name
      character*(*)   set_name

*************************************************************************

C     Move input string to upper case
      CALL upnsquish (set_name,test_string,end)

C     LOCATE NAME IN CALENDAR TABLE
      DO 10 ndx = 1, max_calendars
         IF (test_string(1:3) .EQ. allowed_calendars(ndx)(1:3)) 
     .        GOTO 100
 10   END DO

C     error: calendar not found

      status = merr_varlim
      goto 1000

C     calendar found, set current calendar

c 100  cal_name = test_string(1:end)

 100  cal_name = allowed_calendars(ndx)

      CALL tm_load_calendar (cal_name, status)

 1000 RETURN

      END

