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
* Return memory array parameters/indices for a string data array
* described by datnam
*
* Input:
*     datnam - description of the data array to retrieve
*     lennam - actual length of datnam
*
* Output:
*     arraystart - offset into memory giving the start of the data array
*     memlo, memhi - array dimensions
*                    (memlo() = memhi() = -999 for invalid axes)
*     steplo, stephi, incr - step values for the actual data requested
*                    (steplo() = stephi() = -999 for invalid axes; incr always 1)
*     axtyp - AXISTYPE parameter values describing the axes
*     errmsg - error message if an error occurs
*     lenerr - actual length of errmsg, will be zero if and only if no errors
*
      SUBROUTINE GET_STR_DATA_ARRAY_PARAMS(datnam, lennam, arraystart,
     .                          memlo, memhi, steplo, stephi, 
     .                          incr, axtyp, errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xerrmsg_text.cmn'
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'pyferret.parm'

*     Passed arguments
      TYPE(mem_table_slot) :: arraystart
      CHARACTER*(*) datnam, errmsg
      INTEGER       lennam, lenerr, 
     .              memlo(nferdims), memhi(nferdims),
     .              steplo(nferdims), stephi(nferdims),
     .              incr(nferdims), axtyp(nferdims)

*     Function declarations
      INTEGER TM_LENSTR
      LOGICAL GEOG_LABEL

*     Local variables
      INTEGER sts, mr, cx, k, cmnd_stack_level, grid, line

*     Use GET_FER_COMMAND with a LOAD command to deal with parsing the data description
      CALL GET_FER_COMMAND('LOAD ' // datnam(1:lennam), sts, *1000)

*     Get the data into memory
      CALL GET_CMND_DATA(cx_last, ptype_string, sts)
      IF ( sts .NE. FERR_OK ) THEN
           GOTO 1000
      ENDIF

      mr = is_mr(isp)
      cx = is_cx(isp)

      arraystart = memry(mr)

*     Step values for this array.
*     If the whole array was not requested, a new copy of the data
*     has been made in memory with unit increments (or so it appears).
      DO 20 k = 1,nferdims
          memlo(k) = mr_lo_ss(mr,k)
          memhi(k) = mr_hi_ss(mr,k)
          steplo(k) = cx_lo_ss(cx,k)
          stephi(k) = cx_hi_ss(cx,k)
          incr(k) = 1
   20 CONTINUE

*     Axis types
      grid = cx_grid(cx)
      IF ( grid .EQ. unspecified_int4 ) THEN
          errmsg = 'Unexpected error: no grid found'
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      DO 30 k = 1,nferdims
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

