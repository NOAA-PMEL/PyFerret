      SUBROUTINE TM_SET_CURRENT_CALENDAR (set_name, status)

C     Set the current calendar name and id in common current_calendar
C     and loads calendar info into common.
C
C     Returns merr_varlim if calendar is invalid, merr_ok successful completion
C     J Davison 10.99
C     V 1.0
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

#ifdef unix
      include 'calendar.cmn'
      include 'tmap_errors.parm'
#else
      INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif
      integer         end,ndx,tm_get_calendar_id, status
      character*255   test_string

      character*(*)   set_name

      external        tm_get_calendar_id

*************************************************************************

C     MOVE INPUT STRING TO UPPER CASE
      call upnsquish (set_name,test_string,end)

C     LOCATE NAME IN CALENDAR TABLE
      do 10 ndx = 1, max_calendars
         if (test_string(1:end) .eq. calendar_names(ndx))
     .        goto 100
 10   end do

C     ERROR: CALENDAR NOT FOUND
      status = merr_varlim
      goto 1000

C     CALENDAR FOUND, SET CURRENT CALENDAR
 100  cal_name = test_string(1:end)
      cal_id   = tm_get_calendar_id (cal_name)

      call tm_load_calendar (cal_id, status)

 1000 return

      end

