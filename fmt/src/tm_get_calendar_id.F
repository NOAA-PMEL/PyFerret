      INTEGER FUNCTION TM_GET_CALENDAR_ID (calendar_name)

C     Return the calendar id associated with calendar_name, a calendar name
C     Zero is returned if calendar_name is not a valid calendar.
C     J Davison 10.99
C     V 1.0
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

#ifdef unix
      include 'calendar.cmn'
#else
      INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif

      integer         end,ndx
      character*255   test_string

      character*(*)   calendar_name

*************************************************************************

C     MOVE INPUT STRING TO UPPER CASE
      call upnsquish (calendar_name,test_string,end)

C     LOCATE NAME IN CALENDAR TABLE
      do 10 ndx = 1, max_calendars
         if (test_string(1:end) .eq. calendar_names(ndx))
     .        goto 100
 10   end do

C     ERROR: CALENDAR NOT FOUND
      tm_get_calendar_id = 0
      goto 1000

C     CALENDAR FOUND, RETURN ID
 100  tm_get_calendar_id = ndx

 1000 return

      end
