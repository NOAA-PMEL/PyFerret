	LOGICAL FUNCTION NO_LINE_RANGE( array, npts, bad_flag, val1 )

*
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*
*
* determine if all the data indicated are of the same value
* if true val1=bad_flag if all points are bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/88
* revision 0.1 - 9/22/88 - bug fix.  start at subscript 1
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* *acm* 10/13/20 Do the comparisons in single-precision.

        IMPLICIT NONE
      include	'ferret.parm'

* calling argument declarations:
      INTEGER	npts
	    REAL 	array( npts ), bad_flag, val1

* internal variable declarations:
	    INTEGER	i
	    REAL*4 bad_r4, val_r4, arr_r4

* comparison value
      bad_r4 = bad_flag
      val_r4 = bad_r4

* check all data within the indicated context
	    DO 100 i = 1, npts
		     arr_r4 = array(i)
		     IF ( arr_r4 .NE. val_r4 ) THEN
              IF     ( val_r4 .EQ. bad_r4 ) THEN
                 val_r4 = arr_r4
              ELSEIF ( arr_r4 .NE. bad_r4 ) THEN
                 NO_LINE_RANGE = .FALSE.
             val1 = val_r4
                 RETURN
              ENDIF
          ENDIF
  100 CONTINUE

* they were all identical --> val1 contains bad_flag if they were all bad
	    NO_LINE_RANGE = .TRUE.

	    val1 = val_r4
      IF (val_r4 .EQ. bad_r4) val1 = bad_flag

	    RETURN
	    END
