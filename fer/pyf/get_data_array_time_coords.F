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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.



*
* Return time integer array coordinates for an axis of a data array loaded
* and described from a call to GET_DATA_ARRAY_PARAMS
*
* Input:
*     axnum - axis number to return coordinates
*     numcoords - number of coordinates for this axis (for error checking)
*     errmsg - error message if an error occurs
*     lenerr - actual length of errmsg, will be zero if and only if no errors
*
* Output:
*     axcoords - time integer array coordinates
*     caltyp - CALTYPE parameter number identifying the calendar
*     axname - axis name - null terminated
*
      SUBROUTINE GET_DATA_ARRAY_TIME_COORDS(axcoords, caltyp, axname,
     .                               axnum, numcoords, errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'pyferret.parm'

*     Passed arguments
      CHARACTER*(*) axname, errmsg
      INTEGER       caltyp, axnum, numcoords, lenerr
      INTEGER       axcoords(6,numcoords)

*     Function declarations
      INTEGER TM_LENSTR, TM_GET_CALENDAR_ID
      LOGICAL GEOG_LABEL
      REAL*8  TM_WORLD

*     Local variables
      INTEGER cx, grid, line, ss_low, ss_high, k, q,
     .        day, month, year, hour, minute, second, cal_id
      REAL*8 worldsecs
      CHARACTER*32 timestr, monthstr, calname

      cx = is_cx(isp)
      grid = cx_grid(cx)
      IF ( grid .EQ. unspecified_int4 ) THEN
          errmsg = 'Unexpected error: no grid found'
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF

      line = grid_line(axnum, grid)
      IF ((line .EQ. munknown) .OR. (line .EQ. mnormal)) THEN
          errmsg = 'Unexpected error: unknown or normal axis'
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF

      ss_low = cx_lo_ss(cx, axnum)
      ss_high = cx_hi_ss(cx, axnum)
      IF ( (ss_high - ss_low + 1) .NE. numcoords ) THEN
          errmsg = 'Unexpected error: mismatch of the number of coords'
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      DO 100 k = ss_low,ss_high
          worldsecs = TM_WORLD(k, grid, axnum, box_middle)
          CALL TSTEP_TO_DATE_OLD(grid, worldsecs, 6, timestr)
*         Try to read as DD-MTH-YYYY HH:MM:SS
*         If fails, try another format
          READ(timestr, FMT=110, ERR=20) day, monthstr, year,
     .                                   hour, minute, second
          GOTO 90
*         Try to read as DD-MTH HH:MM:SS
*         If fails, report error
   20     READ(timestr, FMT=120, ERR=500) day, monthstr, 
     .                                    hour, minute, second
          year = 0
   90     CONTINUE
*         Convert month string to a number
          IF ( monthstr .EQ. 'JAN' ) THEN
              month = 1
          ELSE IF ( monthstr .EQ. 'FEB' ) THEN
              month = 2
          ELSE IF ( monthstr .EQ. 'MAR' ) THEN
              month = 3
          ELSE IF ( monthstr .EQ. 'APR' ) THEN
              month = 4
          ELSE IF ( monthstr .EQ. 'MAY' ) THEN
              month = 5
          ELSE IF ( monthstr .EQ. 'JUN' ) THEN
              month = 6
          ELSE IF ( monthstr .EQ. 'JUL' ) THEN
              month = 7
          ELSE IF ( monthstr .EQ. 'AUG' ) THEN
              month = 8
          ELSE IF ( monthstr .EQ. 'SEP' ) THEN
              month = 9
          ELSE IF ( monthstr .EQ. 'OCT' ) THEN
              month = 10
          ELSE IF ( monthstr .EQ. 'NOV' ) THEN
              month = 11
          ELSE IF ( monthstr .EQ. 'DEC' ) THEN
              month = 12
          ELSE
              GOTO 500
          ENDIF
          q = k - ss_low + 1
          axcoords(TIMEARRAY_DAYINDEX,q) = day
          axcoords(TIMEARRAY_MONTHINDEX,q) = month
          axcoords(TIMEARRAY_YEARINDEX,q) = year
          axcoords(TIMEARRAY_HOURINDEX,q) = hour
          axcoords(TIMEARRAY_MINUTEINDEX,q) = minute
          axcoords(TIMEARRAY_SECONDINDEX,q) = second
  100 CONTINUE
  110 FORMAT(I2,X,A3,X,I4,X,I2,X,I2,X,I2)
  120 FORMAT(I2,X,A3,X,I2,X,I2,X,I2)

      calname = line_cal_name(line)
      cal_id = TM_GET_CALENDAR_ID(calname)
      IF ( cal_id .EQ. gregorian ) THEN
          caltyp = CALTYPE_GREGORIAN
      ELSE IF ( cal_id .EQ. noleap ) THEN
          caltyp = CALTYPE_NOLEAP
      ELSE IF ( cal_id .EQ. julian ) THEN
          caltyp = CALTYPE_JULIAN
      ELSE IF ( cal_id .EQ. d360 ) THEN
          caltyp = CALTYPE_360DAY
      ELSE IF ( cal_id .EQ. all_leap ) THEN
          caltyp = CALTYPE_ALLLEAP
      ELSE
          caltyp = CALTYPE_NONE
      ENDIF

      k = TM_LENSTR(line_name(line))
      IF ( k .GT. 0 ) THEN
          axname = line_name(line)(1:k) // CHAR(0)
      ELSE
          axname = CHAR(0)
      ENDIF

      errmsg = ' '
      lenerr = 0
      RETURN

  500 errmsg = 'Unexpected date string: ' // timestr
      lenerr = TM_LENSTR(errmsg)
      RETURN

      END

