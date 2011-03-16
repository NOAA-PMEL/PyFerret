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



*     Assign axisnum, axisstart, and axisend with the number, start
*     index, and end index of a line, either existing or created by
*     this routine, that matches the rest of the axis information
*     arguments.  An empty (blank) axisname will match all names of
*     existing axes.  If an error occurs, errmsg will be assigned
*     with the error message and lenerrmsg will be the length of that
*     message (a positive value).  If successful, lenerrmsg will be
*     zero on return.
*         axiscoords is a array of floating-point axis coordinates

      SUBROUTINE GET_AXIS_NUM(axisnum, axisstart, axisend,
     .                        axisname, axisunit, axiscoords,
     .                        numcoords, axistype, errmsg, lenerrmsg)
      IMPLICIT NONE

#include "gt_lib.parm"
#include "tmap_dset.parm"
      INCLUDE 'implicit.parm'
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'tmap_errors.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'ferret_lib.parm'

*     Passed arguments
      CHARACTER*(*) axisname, axisunit, errmsg
      INTEGER       axisnum, axisstart, axisend,
     .              numcoords, axistype, lenerrmsg
      REAL*8        axiscoords
      DIMENSION     axiscoords(numcoords)

*     Function declarations
      INTEGER TM_LENSTR, TM_UNIT_ID, STR_UPCASE
      LOGICAL TM_LEGAL_NAME, CHECK_LINE_SUBSET

*     Local arguments
      CHARACTER*(64) capaxname
      INTEGER        axucode, k, sts, first_pt, j
      REAL*8         delta, eps, value, modulen
      LOGICAL        reg, modu

*     Sanity check
      IF ( numcoords .LT. 1 ) THEN
          errmsg = 'Non-positive numcoords passed to GET_AXIS_NUM'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     Uppercase the name for comparisons
      IF ( axisname .NE. '' ) THEN
          sts = STR_UPCASE(capaxname, axisname)
          IF ( .NOT. TM_LEGAL_NAME(capaxname) ) THEN
              errmsg = 'Invalid axis name given'
              lenerrmsg = TM_LENSTR(errmsg)
              RETURN
          ENDIF
      ELSE
          capaxname = ''
      ENDIF

*     Get the code number of known units; returns zero for unknown units
      axucode = TM_UNIT_ID(axisunit)

      IF ( numcoords .GT. 1 ) THEN
          delta = (axiscoords(numcoords) - axiscoords(1)) /
     .                       (numcoords - 1)
      ELSE
          delta = 0.0
      ENDIF
      eps = abs(delta) * 1.0E-7
      IF ( eps .LT. 1.0E-14 ) THEN
          eps = 1.0E-14
      ENDIF
*     Check for regular coordinates
      reg = .TRUE.
      DO 10 k = 2, numcoords-1
         value = axiscoords(1) + (k-1) * delta
         IF ( abs(value - axiscoords(k)) .GT. eps ) THEN
             reg = .FALSE.
             GOTO 20
         ENDIF
   10 CONTINUE

*     Crude assumption:
*        longitude are modular with a modulo of 360 deg
*        and nothing else is modular.
   20 IF ( axistype .EQ. AXISTYPE_LONGITUDE ) THEN
          modu = .TRUE.
          modulen = 360.0
      ELSE
          modu = .FALSE.
          modulen = 0.0
      ENDIF

*     See if a dynamic line already exists containing the provided info
      k = max_lines
   50 k = line_flink(k)
      IF ( k .LE. max_lines ) THEN
          GOTO 100
      ENDIF
      IF ( line_class(k) .EQ. pline_class_basic ) THEN
          IF ( CHECK_LINE_SUBSET(k, axisstart, axisend, capaxname,
     .                    axisunit, axiscoords, numcoords, delta, eps,
     .                    reg, modu, modulen, axucode) ) THEN
              axisnum = k
              GOTO 500
          ENDIF
      ENDIF
      GOTO 50

*     See if a permanent line already exists containing the provided info
 100  CONTINUE
      DO 110 k = 1, max_lines
          IF ( (line_class(k) .EQ. pline_class_basic) .AND.
     .         (line_name(k) .NE. char_init16) ) THEN
              IF ( CHECK_LINE_SUBSET(k, axisstart, axisend, capaxname,
     .                   axisunit, axiscoords, numcoords, delta, eps,
     .                   reg, modu, modulen, axucode) ) THEN
                  axisnum = k
                  GOTO 500
              ENDIF
          ENDIF
  110 CONTINUE

*     If we got here, there is no line containing the provided info.
*     So create a dynamic line.
  150 CALL TM_ALLO_DYN_LINE(k, sts)
      IF ( sts .NE. merr_ok ) THEN
          errmsg = 'No space left for a new dynamic line'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      IF ( capaxname .EQ. '' ) THEN
          WRITE(line_name(k), 160) (k - max_lines)
  160     FORMAT('(AX',I3.3,')')
          line_name_orig(k) = line_name(k)
      ELSE
          line_name(k)      = capaxname
          line_name_orig(k) = axisname
      ENDIF
      line_units(k)         = axisunit
      line_unit_code(k)     = axucode
      line_t0(k)            = char_init20
      line_shift_origin(k)  = .FALSE.
      line_tunit(k)         = 0.0
      line_parent(k)        = 0
      line_class(k)         = pline_class_basic
      line_modulo(k)        = modu
      line_modulo_len(k)    = modulen
      IF ( axistype .EQ. AXISTYPE_LONGITUDE ) THEN
          line_direction(k) = axis_orients(1)
      ELSE IF ( axistype .EQ. AXISTYPE_LATITUDE ) THEN
          line_direction(k) = axis_orients(2)
      ELSE IF ( axistype .EQ. AXISTYPE_LEVEL ) THEN
          line_direction(k) = axis_orients(3)
      ELSE
          line_direction(k) = 'NA'
      ENDIF
      line_dim(k)           = numcoords
      line_regular(k)       = reg
      IF ( reg ) THEN
          line_subsc1(k)    = unspecified_int4
          line_start(k)     = axiscoords(1)
          line_delta(k)     = delta
      ELSE
          first_pt          = next_line_mem_pos
          next_line_mem_pos = first_pt + 2 * numcoords + 1
          IF ( next_line_mem_pos .GT. maxlinestore ) THEN
              CALL TM_DEALLO_DYN_LINE(k)
              errmsg = 'No space left for a new irregular line'
              lenerrmsg = TM_LENSTR(errmsg)
              RETURN
          ENDIF
*         Store the coordinates
          DO 170 j = 1, numcoords
              line_mem(first_pt + j - 1) = axiscoords(j)
  170     CONTINUE
*         Compute and store the box boundries
          line_mem(first_pt + numcoords) =
     .         axiscoords(1) - 0.5 * (axiscoords(2) - axiscoords(1))
          DO 180 j = 2, numcoords
              line_mem(first_pt + numcoords + j - 1) =
     .            0.5 * (axiscoords(j) + axiscoords(j-1))
  180     CONTINUE
          line_mem(first_pt + 2 * numcoords) = axiscoords(numcoords) +
     .         0.5 * (axiscoords(numcoords) - axiscoords(numcoords-1))
*         Assign the line info
          line_subsc1(k)    = first_pt
          line_start(k)     = axiscoords(1)
          line_delta(k)     = unspecified_val8
      ENDIF
      line_use_cnt(k)       = 0

      axisnum = k
      axisstart = 1
      axisend = numcoords

  500 lenerrmsg = 0
      RETURN

      END



*     Assign axisnum, axisstart, and axisend with the number, start
*     index, and end index of a line, either existing or created by
*     this routine, that matches the rest of the axis information
*     arguments.  An empty (blank) axisname will match all names of
*     existing axes.  If an error occurs, errmsg will be assigned
*     with the error message and lenerrmsg will be the length of that
*     message (a positive value).  If successful, lenerrmsg will be
*     zero on return.
*         axiscoords is a array of 6 integer time values per coordinate

      SUBROUTINE GET_TIME_AXIS_NUM(axisnum, axisstart, axisend,
     .                             axisname, caltype, axiscoords,
     .                             numcoords, errmsg, lenerrmsg)
      IMPLICIT NONE

#include "gt_lib.parm"
#include "tmap_dset.parm"
      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'
      INCLUDE 'implicit.parm'
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'tmap_errors.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xunits.cmn_text'
      EXTERNAL XUNITS_DATA
      INCLUDE 'ferret_lib.parm'

*     Passed arguments
      CHARACTER*(*) axisname, errmsg
      INTEGER       axisnum, axisstart, axisend, caltype,
     .              axiscoords, numcoords, lenerrmsg
      DIMENSION     axiscoords(6,numcoords)

*     Function declarations
      INTEGER TM_LENSTR, TM_UNIT_ID, STR_UPCASE
      LOGICAL TM_LEGAL_NAME, CHECK_TIME_LINE_SUBSET
      REAL*8  TM_SECS_FROM_BC

*     Local arguments
      CHARACTER*(64) capaxname, axisunit
      CHARACTER*(16) calname
      INTEGER        sts, calid, k, axucode, first_pt, j
      REAL*8         modulen, delta, eps, value, base
      LOGICAL        modu, reg

*     Sanity check
      IF ( numcoords .LT. 1 ) THEN
          errmsg = 'Non-positive numcoords passed to GET_TIME_AXIS_NUM'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     Uppercase the name for comparisons
      IF ( axisname .NE. '' ) THEN
          sts = STR_UPCASE(capaxname, axisname)
          IF ( .NOT. TM_LEGAL_NAME(capaxname) ) THEN
              errmsg = 'Invalid axis name given'
              lenerrmsg = TM_LENSTR(errmsg)
              RETURN
          ENDIF
      ELSE
          capaxname = ''
      ENDIF

*     Get the calendar for this axis and initialize axisunit
      IF ( caltype .EQ. CALTYPE_360DAY ) THEN
          calid = d360
          axisunit = 'year360'
      ELSE IF ( caltype .EQ. CALTYPE_NOLEAP ) THEN
          calid = noleap
          axisunit = 'noleap_year'
      ELSE IF ( caltype .EQ. CALTYPE_GREGORIAN ) THEN
          calid = gregorian
          axisunit = 'gregorian_year'
      ELSE IF ( caltype .EQ. CALTYPE_JULIAN ) THEN
          calid = julian
          axisunit = 'julian_year'
      ELSE IF ( caltype .EQ. CALTYPE_ALLLEAP ) THEN
          calid = all_leap
          axisunit = 'year366'
      ELSE
          errmsg = 'Unknown calendar type'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     Figure out the unit for this time series
      DO 10 k = 2, numcoords
         IF ( axiscoords(TIMEARRAY_SECONDINDEX,k) .NE.
     .        axiscoords(TIMEARRAY_SECONDINDEX,1) ) THEN
             axisunit = 'sec'
             GOTO 100
         ENDIF
   10 CONTINUE
      DO 20 k = 2, numcoords
         IF ( axiscoords(TIMEARRAY_MINUTEINDEX,k) .NE.
     .        axiscoords(TIMEARRAY_MINUTEINDEX,1) ) THEN
             axisunit = 'min'
             GOTO 100
         ENDIF
   20 CONTINUE
      DO 30 k = 2, numcoords
         IF ( axiscoords(TIMEARRAY_HOURINDEX,k) .NE.
     .        axiscoords(TIMEARRAY_HOURINDEX,1) ) THEN
             axisunit = 'hour'
             GOTO 100
         ENDIF
   30 CONTINUE
      DO 40 k = 2, numcoords
         IF ( axiscoords(TIMEARRAY_DAYINDEX,k) .NE.
     .        axiscoords(TIMEARRAY_DAYINDEX,1) ) THEN
             axisunit = 'day'
             GOTO 100
         ENDIF
   40 CONTINUE
      DO 50 k = 2, numcoords
         IF ( axiscoords(TIMEARRAY_MONTHINDEX,k) .NE.
     .        axiscoords(TIMEARRAY_MONTHINDEX,1) ) THEN
*            A "month" unit is ill-defined so Ferret uses days
             axisunit = 'day'
             GOTO 100
         ENDIF
   50 CONTINUE
*     Use the initialized year axisunit
  100 axucode = TM_UNIT_ID(axisunit)
      IF ( (axucode .LT. pun_1st_time) .OR.
     .     (axucode .GT. pun_last_time) ) THEN
          errmsg = 'Unexpected unknown time unit'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     If all years are zero, assume annual modulo axis
      DO 110 k = 1, numcoords
          IF ( axiscoords(TIMEARRAY_YEARINDEX,k) .NE. 0 ) THEN
              modu = .FALSE.
              modulen = 0
              GOTO 150
          ENDIF
  110 CONTINUE
      modu = .TRUE.
      modulen = cals_yeardays(calid) * (24.0 * 60.0 * 60.0) /
     .                  un_convert(axucode)

*     Temporarily grab memory for floating-point time coordinates
  150 first_pt = next_line_mem_pos
      IF ( first_pt + numcoords .GT. maxlinestore ) THEN
          errmsg = 'No memory for floating-point time coordiantes'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      DO 160 k = 1, numcoords
          line_mem(first_pt + k - 1) = TM_SECS_FROM_BC(calid,
     .                        axiscoords(TIMEARRAY_YEARINDEX,k),
     .                        axiscoords(TIMEARRAY_MONTHINDEX,k),
     .                        axiscoords(TIMEARRAY_DAYINDEX,k),
     .                        axiscoords(TIMEARRAY_HOURINDEX,k),
     .                        axiscoords(TIMEARRAY_MINUTEINDEX,k),
     .                        axiscoords(TIMEARRAY_SECONDINDEX,k),
     .                        sts) / un_convert(axucode)
  160 CONTINUE
      IF ( numcoords .GT. 1 ) THEN
          delta = (line_mem(first_pt + numcoords - 1) -
     .             line_mem(first_pt)) / (numcoords - 1)
      ELSE
          delta = 0.0
      ENDIF
      eps = abs(delta) * 1.0E-7
      IF ( eps .LT. 1.0E-14 ) THEN
          eps = 1.0E-14
      ENDIF
*     Check for regular coordinates
      reg = .TRUE.
      DO 170 k = 2, numcoords-1
         value = line_mem(first_pt) + (k-1) * delta
         IF ( abs(value - line_mem(first_pt + k - 1)) .GT. eps ) THEN
             reg = .FALSE.
             GOTO 200
         ENDIF
  170 CONTINUE

*     See if a dynamic line already exists containing the provided info
  200 k = max_lines
  210 k = line_flink(k)
      IF ( k .LE. max_lines ) THEN
          GOTO 300
      ENDIF
      IF ( (line_class(k) .EQ. pline_class_basic) .AND.
     .     (line_t0(k) .NE. char_init20) ) THEN
          IF ( CHECK_TIME_LINE_SUBSET(k, axisstart, axisend, capaxname,
     .                    calid, line_mem(first_pt), numcoords, delta,
     .                    eps, reg, modu, modulen, axucode) ) THEN
              axisnum = k
              GOTO 500
          ENDIF
      ENDIF
      GOTO 210

*     See if a permanent line already exists containing the provided info
 300  CONTINUE
      DO 310 k = 1, max_lines
          IF ( (line_class(k) .EQ. pline_class_basic) .AND.
     .         (line_name(k) .NE. char_init16) .AND.
     .         (line_t0(k) .NE. char_init20) ) THEN
              IF ( CHECK_TIME_LINE_SUBSET(k, axisstart, axisend,
     .                   capaxname, calid, line_mem(first_pt),
     .                   numcoords, delta, eps, reg, modu, modulen,
     .                   axucode) ) THEN
                  axisnum = k
                  GOTO 500
              ENDIF
          ENDIF
  310 CONTINUE

*     If we got here, there is no line containing the provided info.
*     So create a dynamic line.
  350 CALL TM_ALLO_DYN_LINE(k, sts)
      IF ( sts .NE. merr_ok ) THEN
          errmsg = 'No space left for a new dynamic time line'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      IF ( capaxname .EQ. '' ) THEN
          WRITE(line_name(k), 360) (k - max_lines)
  360     FORMAT('(AX',I3.3,')')
          line_name_orig(k) = line_name(k)
      ELSE
          line_name(k)      = capaxname
          line_name_orig(k) = axisname
      ENDIF
      line_units(k)         = axisunit
      line_unit_code(k)     = axucode
      IF ( modu ) THEN
          line_t0(k)        = '01-JAN-0000 00:00:00'
          base = 0.0
      ELSE
          line_t0(k)        = '01-JAN-1900 00:00:00'
          base = TM_SECS_FROM_BC(calid, 1900, 1, 1, 0, 0, 0, sts)
      ENDIF
      line_cal_name(k)      = allowed_calendars(calid)
      line_shift_origin(k)  = .FALSE.
      line_tunit(k)         = un_convert(axucode)
      line_parent(k)        = 0
      line_class(k)         = pline_class_basic
      line_modulo(k)        = modu
      line_modulo_len(k)    = modulen
      line_direction(k)     = 'TI'
      line_dim(k)           = numcoords
      line_regular(k)       = reg
      IF ( reg ) THEN
          line_subsc1(k)    = unspecified_int4
          line_start(k)     = line_mem(first_pt) - base
          line_delta(k)     = delta
      ELSE
*         first_pt already assigned
          next_line_mem_pos = first_pt + 2 * numcoords + 1
          IF ( next_line_mem_pos .GT. maxlinestore ) THEN
              CALL TM_DEALLO_DYN_LINE(k)
              errmsg = 'No space left for a new irregular time line'
              lenerrmsg = TM_LENSTR(errmsg)
              RETURN
          ENDIF
*         Modify the coordinates to the new t0 baseline
          DO 370 j = 1, numcoords
  370         line_mem(first_pt+j-1) = line_mem(first_pt+j-1) - base
*         Compute and store the box boundries
          line_mem(first_pt + numcoords) = line_mem(first_pt) - 0.5 *
     .                   (line_mem(first_pt + 1) - line_mem(first_pt))
          DO 380 j = 2, numcoords
              line_mem(first_pt + numcoords + j - 1) = 0.5 *
     .        (line_mem(first_pt + j - 1) + line_mem(first_pt + j- 2))
  380     CONTINUE
          line_mem(first_pt + 2 * numcoords) =
     .         line_mem(first_pt + numcoords - 1) +
     .         0.5 * (line_mem(first_pt + numcoords - 1) -
     .                line_mem(first_pt + numcoords - 2))
*         Assign the line info
          line_subsc1(k)    = first_pt
          line_start(k)     = line_mem(first_pt)
          line_delta(k)     = unspecified_val8
      ENDIF
      line_use_cnt(k)       = 0

      axisnum = k
      axisstart = 1
      axisend = numcoords

  500 lenerrmsg = 0
      RETURN

      END

