      INTEGER FUNCTION TM_GET_CALENDAR_ID (cal_name)

C     Return the calendar id associated with cal_name, a calendar name
C     Zero is returned if cal_name is not a valid calendar.
C     J Davison 10.99
C     V 1.0
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

C 2/13/01  ACM  test for the calendar names:
C	"STANDARD" which is equivalent to 'GREGORIAN'
C	"COMMON_YEAR" for "NOLEAP"
C	"D360"  or "360" for "360DAY"

#ifdef unix
      include 'calendar.decl'
      include 'calendar.cmn'
#else
      INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif

      integer         end,ndx
      character*255   test_string

      character*(*)   cal_name

*************************************************************************

C     MOVE INPUT STRING TO UPPER CASE
      call upnsquish (cal_name,test_string,end)

C  Allow STANDARD same calendar as GREGORIAN
      IF (test_string(1:3) .EQ. 'STA')  test_string = 'GREGORIAN'
      IF (test_string(1:3) .EQ. 'COM')  test_string = 'NOLEAP'
      IF (test_string(1:3) .EQ. 'D36')  test_string = '360DAY'

C     LOCATE NAME IN CALENDAR TABLE
      do 10 ndx = 1, max_calendars
         if (test_string(1:3) .eq. allowed_calendars(ndx)(1:3))
     .        goto 100
 10   end do

C     ERROR: CALENDAR NOT FOUND
      tm_get_calendar_id = 0
      goto 1000

C     CALENDAR FOUND, RETURN ID
 100  tm_get_calendar_id = ndx

 1000 return

      end
