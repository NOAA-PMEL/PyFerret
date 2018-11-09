      SUBROUTINE modscat(axmin, axmax, axmod, nscat, scatm)
*
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
* ACM 11/00
*  Called by scat2grid* gridding functions.
*  If there is a modulo axis, apply modulo-ness to the coordinates in that 
*  direction of the scattered points.
*
* *kms* 4/2012 Modified to pass the min, max, and modulo for the axis
*              If axmin and axmax are same location after modulo, axmax
*              values will be returned as axmin.
*
*  axmin        minimum value for the modulo axis
*  axmax        maximum value for the modulo axis
*  axmod        modulo length for the modulo axis
*  nscat        number of scattered points
*  scatm        input: scattered points coordinate array for the modulo axis
*               output: updated coordinates in the modulo range (if possible)

      IMPLICIT NONE
*  Input arguments
      REAL*8 axmin, axmax, axmod
      INTEGER nscat

*  Input/output arguments
      REAL scatm(*)

*  Local variables
      INTEGER idx
      REAL*8 val

* Check this is a valid modulo axis
      IF ( (axmin .GE. axmax) .OR. (axmod .LE. 0.0D0) ) THEN
         RETURN
      ENDIF

* Convert values
      DO 100 idx = 1, nscat
*        Do the conversion in double precision
         val = scatm(idx)
         DO WHILE ( val .GE. axmax )
            val = val - axmod
         ENDDO
         DO WHILE ( val .LT. axmin )
            val = val + axmod
         ENDDO
*        Save the new value if in range
         IF ( (val .GE. axmin) .AND. (val .LE. axmax) ) THEN
            scatm(idx) = val
         ENDIF
  100 CONTINUE

      RETURN
      END

