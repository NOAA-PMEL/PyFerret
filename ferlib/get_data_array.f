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
* Return memory array parameters/indices for a float data array
* described by datnam
*
* Input:
*     datnam - description of the data array to retrieve
*     lennam - actual length of datnam
*     memory - ferret memory
*
* Output:
*     start - offset into memory giving the start of the data array
*     memlo, memhi - array dimensions
*                    (memlo() = memhi() = -999 for invalid axes)
*     steplo, stephi, incr - step values for the actual data requested
*                    (steplo() = stephi() = -999 for invalid axes; incr always 1)
*     datunit - units of the data
*     axtyp - AXISTYPE parameter values describing the axes
*     badflg - value of the bad-data-flag for this data
*     errmsg - error message if an error occurs
*     lenerr - actual length of errmsg, will be zero if and only if no errors
*
      SUBROUTINE GET_DATA_ARRAY_PARAMS(datnam, lennam, memory, start,
     .                          memlo, memhi, steplo, stephi, incr,
     .                          datunit, lendatunit, axtyp, badflg,
     .                          errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xerrmsg_text.cmn'
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xvariables.cmn'

*     MAX_FERRET_NDIM parameter value (must match that in ferret_lib.h)
      INTEGER MAX_FERRET_NDIM
      PARAMETER(MAX_FERRET_NDIM = 4)

*     Passed arguments
      CHARACTER*(*) datnam, datunit, errmsg
      INTEGER       lennam, start, lenerr, lendatunit,
     .              memlo(MAX_FERRET_NDIM), memhi(MAX_FERRET_NDIM),
     .              steplo(MAX_FERRET_NDIM), stephi(MAX_FERRET_NDIM),
     .              incr(MAX_FERRET_NDIM), axtyp(MAX_FERRET_NDIM)
      REAL          memory(*), badflg

*     AXISTYPE parameter values (must match those in ferret_lib.h)
      INTEGER    AXISTYPE_LONGITUDE, AXISTYPE_LATITUDE, AXISTYPE_LEVEL,
     .           AXISTYPE_TIME, AXISTYPE_CUSTOM, AXISTYPE_ABSTRACT,
     .           AXISTYPE_NORMAL
      PARAMETER (AXISTYPE_LONGITUDE = 1,
     .           AXISTYPE_LATITUDE = 2,
     .           AXISTYPE_LEVEL = 3,
     .           AXISTYPE_TIME = 4,
     .           AXISTYPE_CUSTOM = 5,
     .           AXISTYPE_ABSTRACT = 6,
     .           AXISTYPE_NORMAL = 7)

*     Function declarations
      INTEGER TM_LENSTR
      LOGICAL GEOG_LABEL
      CHARACTER*(64) VAR_UNITS

*     Local variables
      INTEGER sts, mr, cx, k, cmnd_stack_level, grid, line

*     Use GET_FER_COMMAND with a LOAD command to deal with parsing the data description
      CALL GET_FER_COMMAND(memory, 'LOAD ' // datnam(1:lennam), sts, *1000)

*     Get the data into memory
      CALL GET_CMND_DATA(memory, cx_last, ptype_float, sts)
      IF ( sts .NE. FERR_OK ) THEN
           GOTO 1000
      ENDIF

      mr = is_mr(isp)
      cx = is_cx(isp)

*     Starting place in memory for this array
      start = (mr_blk1(mr) - 1) * mem_blk_size

*     Step values for this array.
*     If the whole array was not requested, a new copy of the data
*     has been made in memory with unit increments (or so it appears).
      DO 20 k = 1,MAX_FERRET_NDIM
          memlo(k) = mr_lo_ss(mr,k)
          memhi(k) = mr_hi_ss(mr,k)
          steplo(k) = cx_lo_ss(cx,k)
          stephi(k) = cx_hi_ss(cx,k)
          incr(k) = 1
   20 CONTINUE

*     Units of the data
      datunit = VAR_UNITS(cx)
      lendatunit = TM_LENSTR(datunit)

*     Axis types
      grid = cx_grid(cx)
      IF ( grid .EQ. unspecified_int4 ) THEN
          errmsg = 'Unexpected error: no grid found'
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      DO 30 k = 1,MAX_FERRET_NDIM
          IF ( GEOG_LABEL(k, grid) ) THEN
*             In Ferret, if a special {longitude,latitude,level,time} axis,
*             they have to be axis {1,2,3,4}.
*             Do not do axtype(k) = k in case the parameter values change.
              IF ( k .EQ. 1 ) THEN
                  axtyp(k) = AXISTYPE_LONGITUDE
              ELSE IF ( k .EQ. 2 ) THEN
                  axtyp(k) = AXISTYPE_LATITUDE
              ELSE IF ( k .EQ. 3 ) THEN
                  axtyp(k) = AXISTYPE_LEVEL
              ELSE IF ( k .EQ. 4 ) THEN
                  axtyp(k) = AXISTYPE_TIME
              ELSE
                  errmsg = 'Unexpected error: unknown geographical axis'
                  lenerr = TM_LENSTR(errmsg)
                  RETURN
              ENDIF
          ELSE
*             Either custom (has units), abstract (integers without units), or normal to this data
              line = grid_line(k,grid)
              IF ((line .EQ. mnormal) .OR. (line .EQ. munknown)) THEN
                  axtyp(k) = AXISTYPE_NORMAL
              ELSE IF ( line_unit_code(line) .NE. 0 ) THEN
                  axtyp(k) = AXISTYPE_CUSTOM
              ELSE IF ( line_units(line) .NE. ' ' ) THEN
                  axtyp(k) = AXISTYPE_CUSTOM
              ELSE
                  axtyp(k) = AXISTYPE_ABSTRACT
              ENDIF
          ENDIF
   30 CONTINUE

*     Bad-data-flag value
      badflg = mr_bad_data(mr)

*     Success
      errmsg = ' '
      lenerr = 0
      RETURN

*     Error return - get message from FER_LAST_ERROR
 1000 CONTINUE
      CALL CLEANUP_LAST_CMND(cmnd_stack_level)
      CALL GETSYM('FER_LAST_ERROR', errmsg, lenerr, sts)
      IF ( (lenerr .EQ. 1) .AND. (errmsg(1:1) .EQ. ' ') ) THEN
          lenerr = 0
      ENDIF
      IF ( lenerr .LE. 0 ) THEN
          errmsg = 'Unable to load ' // datnam(1:lennam)
          lenerr = TM_LENSTR(errmsg)
      ENDIF
      RETURN

      END



*
* Return coordinates for an axis of a data array loaded and described
* from a call to GET_DATA_ARRAY_PARAMS
*
* Input:
*     axnum - axis number to return coordinates
*     numcoords - number of coordinates for this axis (for error checking)
*     errmsg - error message if an error occurs
*     lenerr - actual length of errmsg, will be zero if and only if no errors
*
* Output:
*     axcoords - axis coordinates
*     axunits - axis unit name - null terminated
*     axname - axis name - null terminated
*
      SUBROUTINE GET_DATA_ARRAY_COORDINATES(axcoords, axunits, axname,
     .                          axnum, numcoords, errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xunits.cmn_text'
      INCLUDE 'xvariables.cmn'

*     Passed arguments
      CHARACTER*(*) axunits, errmsg, axname
      INTEGER       axnum, numcoords, lenerr
      REAL*8        axcoords(numcoords)

*     Function declarations
      INTEGER TM_LENSTR
      LOGICAL GEOG_LABEL
      REAL*8  TM_WORLD

*     Local variables
      INTEGER cx, grid, line, ss_low, ss_high, k, q

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
      DO 10 k = ss_low,ss_high
          q = k - ss_low + 1
          axcoords(q) = TM_WORLD(k, grid, axnum, box_middle)
   10 CONTINUE

      IF ( ((axnum .EQ. 1) .OR. (axnum .EQ. 2)) .AND.
     .      GEOG_LABEL(axnum, grid) ) THEN
*         Ferret standard longitude or latitude axis
*         Set units to match cdms2 defaults for longitude and latitude
          IF ( axnum .EQ. 1 ) THEN
              axunits = 'degrees_east' // CHAR(0)
          ELSE
              axunits = 'degrees_north' // CHAR(0)
          ENDIF
      ELSE
*         Use the stored units string, if assigned
          k = TM_LENSTR(line_units(line))
          IF ( k .GT. 0 ) THEN
              axunits = line_units(line)(1:k) // CHAR(0)
          ELSE
              axunits(1:1) = CHAR(0)
          ENDIF
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

      END



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

*     Passed arguments
      CHARACTER*(*) axname, errmsg
      INTEGER       caltyp, axnum, numcoords, lenerr
      INTEGER       axcoords(6,numcoords)

*     TIMEARRAY_INDEX parameter values (must be one more than those in ferret_lib.h)
      INTEGER TIMEARRAY_DAYINDEX, TIMEARRAY_MONTHINDEX,
     .        TIMEARRAY_YEARINDEX, TIMEARRAY_HOURINDEX,
     .        TIMEARRAY_MINUTEINDEX, TIMEARRAY_SECONDINDEX
      PARAMETER (TIMEARRAY_DAYINDEX  = 1,
     .           TIMEARRAY_MONTHINDEX = 2,
     .           TIMEARRAY_YEARINDEX = 3,
     .           TIMEARRAY_HOURINDEX = 4,
     .           TIMEARRAY_MINUTEINDEX = 5,
     .           TIMEARRAY_SECONDINDEX = 6)
*     CALTYPE parameter values (must match those in ferret_lib.h)
      INTEGER CALTYPE_NONE, CALTYPE_360DAY,
     .        CALTYPE_NOLEAP, CALTYPE_GREGORIAN,
     .        CALTYPE_JULIAN, CALTYPE_ALLLEAP
      PARAMETER (CALTYPE_NONE = -1,
     .           CALTYPE_360DAY = 0,
     .           CALTYPE_NOLEAP = 50000,
     .           CALTYPE_GREGORIAN = 52425,
     .           CALTYPE_JULIAN = 52500,
     .           CALTYPE_ALLLEAP = 60000)

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

