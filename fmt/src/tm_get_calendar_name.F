      CHARACTER*32 FUNCTION TM_GET_CALENDAR_NAME (cal_id)

C     Return the calendar name associated with cal_id an integer code.
C     'no_calendar is returned if cal_id is not a valid code
C     Ansley Manke 1/01

#ifdef unix
      include 'calendar.decl'
      include 'calendar.cmn'
#endif

      INTEGER cal_id

      tm_get_calendar_name = 'NO_CALENDAR'

C     LOCATE NAME IN CALENDAR TABLE
      IF (cal_id .GT. 0   .AND.  cal_id .LT. max_calendars)
     .     tm_get_calendar_name = allowed_calendars(cal_id)

      end
