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



*     Checks if a subsection of superline can describe the axis given
*     by the rest of the arguments other that axisstart and axisend.
*     If so, axisstart and axisend are assigned the first and last
*     index of the coordinates of the line in superline.  A blank
*     (empty) axisname will match any line name.
*     Note: superline is assumed to be a valid line.
*     Note: If superline is exactly the line described, axisstart will
*           be assigned one and axisend will be assigned numcoords.
*
      LOGICAL FUNCTION CHECK_LINE_SUBSET(superline, axisstart, axisend,
     .                               axisname, axisunit, axiscoords,
     .                               numcoords, delta, eps, reg,
     .                               modu, modulen, axucode)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'pyferret.parm'

*     Passed arguments
      INTEGER       superline, axisstart, axisend,
     .              numcoords, axucode
      CHARACTER*(*) axisname, axisunit
      REAL*8        axiscoords, delta, eps, modulen
      LOGICAL       reg, modu
      DIMENSION     axiscoords(numcoords)

*     Local arguments
      INTEGER       i, j, k
      REAL*8        value

*     First check the quick and easy, then go to the strings and coordinates
      k = superline
      IF ( .NOT. reg .EQV. line_regular(k) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( .NOT. modu .EQV. line_modulo(k) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( modu .AND. 
     .     (abs(modulen - line_modulo_len(k)) .GT. eps) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( axucode .NE. line_unit_code(k) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( (axucode .EQ. 0) .AND. (axisunit .NE. line_units(k)) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( (axisname .NE. '') .AND. (axisname .NE. line_name(k)) ) THEN
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
*     Check the coordinates - may be a subset
      IF ( reg ) THEN
*         First check the spacing between coordinates
          IF ( (numcoords .GT. 1) .AND.
     .         (abs(delta - line_delta(k)) .GT. eps) ) THEN
              CHECK_LINE_SUBSET = .FALSE.
              RETURN
          ENDIF
*         Look for the first coordinate
          DO 30 j = 1, line_dim(k)
              value = line_start(k) + (j-1) * line_delta(k)
              IF ( abs(value - axiscoords(1)) .LT. eps ) THEN
*                 Verify the last coordinate is in there
                  i = j + numcoords - 1
                  IF ( i .GT. line_dim(k) ) THEN
                      CHECK_LINE_SUBSET = .FALSE.
                      RETURN
                  ENDIF
*                 Successfully found the regular axis
                  axisstart = j
                  axisend = i
                  CHECK_LINE_SUBSET = .TRUE.
                  RETURN
              ENDIF
  30      CONTINUE
      ELSE
*         Find the first coordinate
          DO 40 j = 1, line_dim(k)
              value = line_mem(line_subsc1(k) + j - 1)
              IF ( abs(value - axiscoords(1)) .LT. eps ) THEN
                  GOTO 50
              ENDIF
   40     CONTINUE
*         First coordinate not found
          CHECK_LINE_SUBSET = .FALSE.
          RETURN
*         First coordinate found; check the rest of the coordinates
   50     IF ( (j + numcoords - 1) .GT. line_dim(k) ) THEN
              CHECK_LINE_SUBSET = .FALSE.
              RETURN
          ENDIF
          DO 60 i = 2, numcoords
              value = line_mem(line_subsc1(k) + j + i - 2)
              IF ( abs(value - axiscoords(i)) .GE. eps ) THEN
                  CHECK_LINE_SUBSET = .FALSE.
                  RETURN
              ENDIF
   60     CONTINUE
*         Successfully found the irregular axis
          axisstart = j
          axisend = i
          CHECK_LINE_SUBSET = .TRUE.
          RETURN
      ENDIF

      CHECK_LINE_SUBSET = .FALSE.
      RETURN

      END



*     Checks if a subsection of superline can describe the axis given
*     by the rest of the arguments other that axisstart and axisend.
*     If so, axisstart and axisend are assigned the first and last
*     index of the coordinates of the line in superline.  A blank
*     (empty) axisname will match any line name.
*     Note: superline is assumed to be a valid line.
*     Note: If superline is exactly the line described, axisstart will
*           be assigned one and axisend will be assigned numcoords.
*     Note: axiscoords are assumed to be in units of time since
*           01-JAN-0000 00:00:00, and adjustments to a different start
*           for superline are made by this function.

      LOGICAL FUNCTION CHECK_TIME_LINE_SUBSET(superline, axisstart,
     .                            axisend, axisname, calid,
     .                            axiscoords, numcoords, delta, eps,
     .                            reg, modu, modulen, axucode)
      IMPLICIT NONE

      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'tmap_errors.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xunits.cmn_text'
      EXTERNAL XUNITS_DATA
      INCLUDE 'pyferret.parm'

*     Passed arguments
      INTEGER       superline, axisstart, axisend,
     .              calid, numcoords, axucode
      CHARACTER*(*) axisname
      REAL*8        axiscoords, delta, eps, modulen
      LOGICAL       reg, modu
      DIMENSION     axiscoords(numcoords)

*     Function declarations
      REAL*8        TM_SECS_FROM_BC

*     Local arguments
      INTEGER       i, j, k, sts,
     .              t0yr, t0mon, t0day, t0hr, t0min, t0sec
      REAL*8        base, value, factor

*     First check the quick and easy, then go to the strings and coordinates
      k = superline
      IF ( .NOT. reg .EQV. line_regular(k) ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( .NOT. modu .EQV. line_modulo(k) ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
*     units can be different as long as they are time units
      IF ( (line_unit_code(k) .LT. pun_1st_time) .OR.
     .     (line_unit_code(k) .GT. pun_last_time) ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
*     Adjustment factor for unit conversion from superline to the described line
      factor = un_convert(line_unit_code(k)) / un_convert(axucode)
*     line_modulo_len not always set for time axes - ignore
*     IF ( modu ) THEN
*         value = line_modulo_len(k) * factor
*         IF ( abs(modulen - value) .GT. eps ) THEN
*             CHECK_LINE_SUBSET = .FALSE.
*             RETURN
*         ENDIF
*     ENDIF
      IF ( allowed_calendars(calid) .NE. line_cal_name(k) ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      IF ( (axisname .NE. '') .AND. (axisname .NE. line_name(k)) ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
*     Adjust for the t0 baseline of the line
      CALL TM_BREAK_DATE(line_t0(k), calid, t0yr, t0mon,
     .                   t0day, t0hr, t0min, t0sec, sts)
      IF ( sts .NE. merr_ok ) THEN
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
      ENDIF
      base = TM_SECS_FROM_BC(calid, t0yr, t0mon, t0day, t0hr, t0min,
     .                    t0sec, sts) / un_convert(line_unit_code(k))
*     Check the coordinates - may be a subset
      IF ( reg ) THEN
*         First check the spacing between coordinates
          IF ( numcoords .GT. 1 ) THEN
              value = line_delta(k) * factor
              IF ( abs(delta - value) .GT. eps ) THEN
                  CHECK_TIME_LINE_SUBSET = .FALSE.
                  RETURN
              ENDIF
          ENDIF
*         Look for the first coordinate
          DO 130 j = 1, line_dim(k)
              value = line_start(k) + (j-1) * line_delta(k) + base
              value = value * factor
              IF ( abs(value - axiscoords(1)) .LT. eps ) THEN
*                 Verify the last coordinate is in there
                  i = j + numcoords - 1
                  IF ( i .GT. line_dim(k) ) THEN
                      CHECK_TIME_LINE_SUBSET = .FALSE.
                      RETURN
                  ENDIF
*                 Successfully found the regular axis
                  axisstart = j
                  axisend = i
                  CHECK_TIME_LINE_SUBSET = .TRUE.
                  RETURN
              ENDIF
 130      CONTINUE
      ELSE
*         Find the first coordinate
          DO 140 j = 1, line_dim(k)
              value = line_mem(line_subsc1(k) + j - 1) + base
              value = value * factor
              IF ( abs(value - axiscoords(1)) .LT. eps ) THEN
                  GOTO 150
              ENDIF
  140     CONTINUE
*         First coordinate not found
          CHECK_TIME_LINE_SUBSET = .FALSE.
          RETURN
*         First coordinate found; check the rest of the coordinates
  150     IF ( (j + numcoords - 1) .GT. line_dim(k) ) THEN
              CHECK_TIME_LINE_SUBSET = .FALSE.
              RETURN
          ENDIF
          DO 160 i = 2, numcoords
              value = line_mem(line_subsc1(k) + j + i - 2) + base
              value = value * factor
              IF ( abs(value - axiscoords(i)) .GE. eps ) THEN
                  CHECK_TIME_LINE_SUBSET = .FALSE.
                  RETURN
              ENDIF
  160     CONTINUE
*         Successfully found the irregular axis
          axisstart = j
          axisend = i
          CHECK_TIME_LINE_SUBSET = .TRUE.
          RETURN
      ENDIF

      CHECK_TIME_LINE_SUBSET = .FALSE.
      RETURN

      END
