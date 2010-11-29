	SUBROUTINE GET_AXIS_FORMAT( lo_val, hi_val, del_val, 
     .     fmt, use_nice )

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
* ACM 5/2005
* set up the format for PPLUS to properly label the axis tics
* Extracted from axis_format.F
* 
* 2/2008 ACM If the lo and hi value on the axis are the same '
* (which may happen in a PLOT/VS plot) we got a divide by zero.

* calling argument declarations:
	CHARACTER	fmt*4
	REAL*8		lo_val, hi_val, del_val
        LOGICAL         use_nice

* internal variable declarations:
	INTEGER		places, ndec
	REAL		res_del
	INTEGER		n_res

* decimal places needed for axis:
* ... sufficient digits to resolve the tic interval differences and
*     still display the largest axis labels
*  res_del	- labels will be accurate enough to resolve this delta value

* Compute n_res; round up (hi-lo)/(2*del_val)  We label half of the tic marks...

        n_res = INT(0.5+ ABS(hi_val - lo_val) / (2.D0*del_val) )
        IF (n_res .EQ. 0) n_res = del_val
	res_del = ABS( hi_val - lo_val ) / n_res
	CALL GET_PREC_DIGITS( hi_val, lo_val, res_del, places, ndec )

* format to use
* Let smaller ranges be labeled with longitude (range 161.123 to 161.130)
*	use_nice = places .LE. 6
	use_nice = places .LE. 7
	IF ( use_nice ) THEN
	   IF ( ndec .EQ. 0 ) THEN
	      fmt = 'I7'
	   ELSE
	      WRITE ( fmt, 3000 ) ndec
 3000	      FORMAT ('F8.',I1)
	   ENDIF
	ENDIF

        RETURN
        END
