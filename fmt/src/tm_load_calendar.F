      SUBROUTINE TM_LOAD_CALENDAR (cal_name, status)

C     Loads data for the calendar given  into common.  These values are
C     accessed by the date/time functions for all calendar-using plots.
C     Returns status = merr_ok success, merr_varlim failure
C
C     NOTE the current calendar name and id are NOT changed: for that see
C     tm_set_current_calendar().  This permits Ferret queries of a particular
C     calendar without losing knowledge of the calendar in use.
C
C     J Davison 10.99
C     V 1.0
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

#ifdef unix
      include 'calendar.decl'
      include 'calendar.cmn'
      include 'tmap_errors.parm'
#else
      INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

      integer         TM_GET_CALENDAR_ID, cal_id,status,i,j,k
      character*(*)   cal_name

*************************************************************************

      cal_id = TM_GET_CALENDAR_ID (cal_name)

C     OK CALENDAR ID?
      if (cal_id .lt. 1 .or. 
     .    cal_id .gt. max_calendars) then 
         status = merr_varlim
         goto 1000
      end if

C     SET NUM_MONTHS AND NUM_DAYS IN CALENDAR
      num_months = cals_num_months(cal_id)
      num_days = cals_num_days(cal_id)

C     SET MONTH NAMES
      do 110 i = 1, num_months
         month_names(i) = cals_month_names(i,cal_id)
 110  end do

C     SET DAYS_IN_MONTH
      do 120 i = 1, num_months
         days_in_month(i) = cals_days_in_month(i,cal_id)
 120  end do

C     SET DAYS_BEFORE_MONTH
      days_before_month(1) = 0
      do 130 i = 2, num_months
         days_before_month(i) = days_before_month(i-1) 
     .                            + days_in_month(i-1)
 130  end do

C     SET MONTH_BY_DAY
      k = 0
      do 150 j = 1, num_months
         do 140 i = 1, days_in_month(j)
            k = k + 1
            month_by_day(k) = j
 140     end do
 150  end do

C  IDENTIFY LOADED CALENDAR
      calendar_id = cal_id

      status = merr_ok

 1000 return

      end


