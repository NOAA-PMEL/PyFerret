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
      SUBROUTINE GET_DATA_ARRAY_COORDS(axcoords, axunits, axname,
     .                          axnum, numcoords, errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'xunits.cmn_text'
      EXTERNAL XUNITS_DATA
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

