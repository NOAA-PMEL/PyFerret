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
*         axiscoords is a array of 6 integer time values per coordinate
C 12/2022 *acm* Code cleanup: Fixes to calls to routine TM_RM_TMP_LINE


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
      INCLUDE 'pyferret.parm'

*     Passed arguments
      CHARACTER*(*) axisname, errmsg
      INTEGER       axisnum, axisstart, axisend, caltype,
     .              axiscoords, numcoords, lenerrmsg
      DIMENSION     axiscoords(6,numcoords)

*     Function declarations
      INTEGER TM_LENSTR, TM_UNIT_ID, STR_UPCASE
      LOGICAL TM_LEGAL_NAME, CHECK_TIME_LINE_SUBSET
      REAL*8  TM_SECS_FROM_BC, GET_LINE_COORD

*     Local arguments
      CHARACTER*(64) capaxname, uniqaxname, axisunit
      CHARACTER*(16) calname
      INTEGER        sts, calid, k, axucode, first_pt, j, q
      REAL*8         modulen, delta, eps, val, base, firstval
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
      DO 10 j = 2, numcoords
         IF ( axiscoords(TIMEARRAY_SECONDINDEX,j) .NE.
     .        axiscoords(TIMEARRAY_SECONDINDEX,1) ) THEN
             axisunit = 'sec'
             GOTO 100
         ENDIF
   10 CONTINUE
      DO 20 j = 2, numcoords
         IF ( axiscoords(TIMEARRAY_MINUTEINDEX,j) .NE.
     .        axiscoords(TIMEARRAY_MINUTEINDEX,1) ) THEN
             axisunit = 'min'
             GOTO 100
         ENDIF
   20 CONTINUE
      DO 30 j = 2, numcoords
         IF ( axiscoords(TIMEARRAY_HOURINDEX,j) .NE.
     .        axiscoords(TIMEARRAY_HOURINDEX,1) ) THEN
             axisunit = 'hour'
             GOTO 100
         ENDIF
   30 CONTINUE
      DO 40 j = 2, numcoords
         IF ( axiscoords(TIMEARRAY_DAYINDEX,j) .NE.
     .        axiscoords(TIMEARRAY_DAYINDEX,1) ) THEN
             axisunit = 'day'
             GOTO 100
         ENDIF
   40 CONTINUE
      DO 50 j = 2, numcoords
         IF ( axiscoords(TIMEARRAY_MONTHINDEX,j) .NE.
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
      DO 110 j = 1, numcoords
          IF ( axiscoords(TIMEARRAY_YEARINDEX,j) .NE. 0 ) THEN
              modu = .FALSE.
              modulen = 0
              GOTO 150
          ENDIF
  110 CONTINUE
      modu = .TRUE.
      modulen = cals_yeardays(calid) * (24.0 * 60.0 * 60.0) /
     .                  un_convert(axucode)
  150 CONTINUE

*     Create a complete irregular axis to store computed coordinates.
*     It might be kept or it might be thrown away.
      IF ( capaxname .NE. '' ) THEN
*         If a name is given, make sure it is unique, or make it unique.
*         Do this first so the newly created dynamic line will not interfere.
          CALL TM_NEW_LINE_NAME(capaxname, uniqaxname)
      ENDIF
*     Now create the dynamic line
      CALL TM_ALLO_DYN_LINE(q, sts)
      IF ( sts .NE. merr_ok ) THEN
          errmsg = 'No space left for a new dynamic time line'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF
*     Assign it a unique name.
      IF ( capaxname .EQ. '' ) THEN
          WRITE(line_name(q), 360) (q - max_lines)
  360     FORMAT('(AX',I3.3,')')
          line_name_orig(q) = line_name(q)
      ELSE
          line_name(q)      = uniqaxname
          line_name_orig(q) = axisname
      ENDIF
*     Assign the rest of the line info
      line_units(q)         = axisunit
      line_unit_code(q)     = axucode
      IF ( modu ) THEN
          line_t0(q)        = '01-JAN-0000 00:00:00'
          base = 0.0
      ELSE
          line_t0(q)        = '01-JAN-1900 00:00:00'
          base = TM_SECS_FROM_BC(calid, 1900, 1, 1, 0, 0, 0, sts)
      ENDIF
      line_cal_name(q)      = allowed_calendars(calid)
      line_shift_origin(q)  = .FALSE.
      line_tunit(q)         = un_convert(axucode)
      line_parent(q)        = 0
      line_class(q)         = pline_class_basic
      line_modulo(q)        = modu
      line_modulo_len(q)    = modulen
      line_direction(q)     = 'TI'
      line_dim(q)           = numcoords
      line_regular(q)       = .FALSE.
      CALL GET_LINE_DYNMEM(numcoords, q, sts)
      IF ( sts .NE. merr_ok ) THEN
          errmsg = 'No memory left for a temporary irregular line'
          lenerrmsg = TM_LENSTR(errmsg)
*         No memory was allocated so set to regular for cleanup
          line_regular(q) = .TRUE.
          CALL TM_RM_TMP_LINE(q, sts)
          RETURN
      ENDIF
      DO 160 j = 1, numcoords
          val = TM_SECS_FROM_BC(calid,
     .                          axiscoords(TIMEARRAY_YEARINDEX,j),
     .                          axiscoords(TIMEARRAY_MONTHINDEX,j),
     .                          axiscoords(TIMEARRAY_DAYINDEX,j),
     .                          axiscoords(TIMEARRAY_HOURINDEX,j),
     .                          axiscoords(TIMEARRAY_MINUTEINDEX,j),
     .                          axiscoords(TIMEARRAY_SECONDINDEX,j),
     .                          sts) / un_convert(axucode)
          CALL PUT_LINE_COORD(linemem(q)%ptr, j, val - base)
  160 CONTINUE
      firstval = GET_LINE_COORD(linemem(q)%ptr, 1)
      line_start(q)     = firstval
      line_delta(q)     = unspecified_val8

      IF ( numcoords .GT. 1 ) THEN
          delta = (GET_LINE_COORD(linemem(q)%ptr, numcoords) -
     .             firstval) / (numcoords-1)
      ELSE
          delta = 0.0
      ENDIF
      eps = abs(delta) * 1.0E-7
      IF ( eps .LT. 1.0E-14 ) THEN
          eps = 1.0E-14
      ENDIF
*     Check for regular coordinates
      reg = .TRUE.
      DO 170 j = 2, numcoords-1
         val = firstval + (j-1) * delta
         IF ( abs(GET_LINE_COORD(linemem(q)%ptr, j) - val) .GT. eps ) THEN
             reg = .FALSE.
             GOTO 200
         ENDIF
  170 CONTINUE

*     See if a dynamic line (other than the one created above) 
*     already exists containing the provided info
  200 k = max_lines
  210 k = line_flink(k)
      IF ( k .LE. max_lines ) THEN
          GOTO 300
      ENDIF
      IF ( k .EQ. q ) THEN
          GOTO 210
      ENDIF
      IF ( (line_class(k) .EQ. pline_class_basic) .AND.
     .     (line_t0(k) .NE. char_init20) ) THEN
          IF ( CHECK_TIME_LINE_SUBSET(k, axisstart, axisend, 
     .           capaxname, calid, linemem(q)%ptr, numcoords, 
     .           delta, eps, reg, modu, modulen, axucode, base) ) THEN
              axisnum = k
*             Do not need the line created above
              CALL TM_RM_TMP_LINE(q, sts)
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
     .           capaxname, calid, linemem(q)%ptr, numcoords, 
     .           delta, eps, reg, modu, modulen, axucode, base) ) THEN
                  axisnum = k
*                 Do not need the line created above
                  CALL TM_RM_TMP_LINE(q, sts)
                  GOTO 500
              ENDIF
          ENDIF
  310 CONTINUE

*     If we got here, there was no line containing the provided info
*     other than the one created above, so use the line created above.
      IF ( reg ) THEN
*         Change to a regular line, freeing memory for stored coordinates
          CALL FREE_LINE_DYNMEM(q)
          line_regular(q)   = .TRUE.
          line_start(q)     = firstval
          line_delta(q)     = delta
      ELSE
*         Already correctly assigned irregular line; just assign the edges
          CALL TM_IRREGAX_MID(linemem(q)%ptr, lineedg(q)%ptr, numcoords)
      ENDIF
      line_use_cnt(q)       = 0

      axisnum = q
      axisstart = 1
      axisend = numcoords

  500 lenerrmsg = 0
      RETURN

      END

