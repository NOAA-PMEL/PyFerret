c*
c* ef_get_axis_calendar.F
c*
c* Ansley Manke * 
c* 5/2006
c*
c*
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

c*
c* This routine returns the calendar name, days in year, number of 
c* months and days in each month of a time axis
c*

      SUBROUTINE EF_GET_AXIS_CALENDAR(id, iarg, calname, yrdays, 
     .               mnths, d_in_month)

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'

* argument declarations
      INTEGER id, iarg
      INTEGER grid, axis, idim

      CHARACTER*(*) calname
      REAL yrdays
      INTEGER mnths, d_in_month(*)

* Internal declarations
      INTEGER TM_GET_CALENDAR_ID, cx_list(EF_MAX_ARGS), cal_id, i

* Initialize
      yrdays = 0.
      mnths = 0.
      DO 100 i = 1, 12
         d_in_month(i) = 0
100   CONTINUE

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL ef_get_cx_list(cx_list)

* Get the grid and if it has a time axis get the calendar info

      grid = cx_grid(cx_list(iarg))
      idim = t_dim
      axis = grid_line(idim, grid) 

      IF (axis .EQ. munknown) THEN
         calname = 'none'

      ELSE IF (axis .EQ. mnormal) THEN
         calname = 'none'

      ELSE
         calname  = line_cal_name(axis)
         cal_id = TM_GET_CALENDAR_ID (calname)
         yrdays = cals_yeardays(cal_id)      
         mnths = cals_num_months(cal_id)

         DO 110 i = 1, mnths
           d_in_month(i) = cals_days_in_month(i,cal_id)
110      CONTINUE

      ENDIF


      RETURN 
      END
