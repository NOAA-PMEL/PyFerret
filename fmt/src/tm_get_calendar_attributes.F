      SUBROUTINE TM_GET_CALENDAR_ATTRIBUTES (cal_id, nmonths, 
     .                 ndays, yeardays, mon_names, d_in_mon, 
     .                 d_before_mon, mon_by_d)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
C     Loads data for the calendar given  into common.  These values are
C     accessed by the date/time functions for all calendar-using plots.
C     Returns status = merr_ok success, merr_varlim failure
C
C     NOTE the current calendar name and id are NOT changed: for that see
C     tm_set_current_calendar().  This permits Ferret queries of a particular
C     calendar without losing knowledge of the calendar in use.
C
C  12/14/00  ACM  (from Jerry's tm_load_calendar)
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files, add disclaimer
* 5/2020 Issue 1906: treat Proleptic-Gregorian identically to Gregorian

C     0.00000000000000000000000000000000000000000000000000000000000000001

      include 'calendar.decl'
      include 'calendar.cmn'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'xtm_grid.cmn_text'

* Argument declarations
      INTEGER cal_id
      INTEGER nmonths, ndays, d_before_mon(*), mon_by_d(*), d_in_mon(*)
      REAL    yeardays
      CHARACTER*3 mon_names(*)

* Local variable declarations
      INTEGER status, i, j, k, nline
      CHARACTER*20  new_t0
 
C     OK CALENDAR ID?
      if (cal_id .lt. 1 .or. 
     .    cal_id .gt. max_calendars) then 
         status = merr_varlim
         goto 1000
      end if

C     SET NMONTHS AND NDAYS IN CALENDAR

      nmonths = cals_num_months(cal_id)

      ndays = cals_num_days(cal_id)

      yeardays = cals_yeardays(cal_id)

C     SET MONTH NAMES

      do 110 i = 1, nmonths
         mon_names(i) = cals_month_names(i,cal_id)
 110  end do

C     SET D_IN_MON

      do 120 i = 1, nmonths
         d_in_mon(i) = cals_days_in_month(i,cal_id)
 120  end do

C     SET D_BEFORE_MON

      d_before_mon(1) = 0
      do 130 i = 2, nmonths
         d_before_mon(i) = d_before_mon(i-1) 
     .                            + d_in_mon(i-1)
 130  end do

C     SET MON_BY_D
      k = 0
      do 150 j = 1, nmonths
         do 140 i = 1, d_in_mon(j)
            k = k + 1
            mon_by_d(k) = j
 140     end do
 150  end do
      if (cal_id .EQ. julian) mon_by_d(k+1) = mon_by_d(k)
 
      status = merr_ok

 1000 return

      end


