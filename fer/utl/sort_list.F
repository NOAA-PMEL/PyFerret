	SUBROUTINE SORT_LIST(n, valid_min, valid_max,
     .			     bad_flag_in, bad_flag_out,
     .			     list, indices, nvalid)


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
* Sort an input list of values returning both the sorted list and the
* map vector (as floats)
* if there are missing values in the index list, sort em as if they are BIG


* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510 *sh* 8/99
* V530 *sh* use sorting routine SHELL for improved performance
* V530 *acm* 1/23/01 replace shell sort with HEAPSORT for even 
*	better performance.
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'errmsg.parm'

* calling argument declarations:
	INTEGER n, valid_min, valid_max, nvalid
	REAL bad_flag_in, bad_flag_out, list(n), indices(n)

* internal variable declarations:
	INTEGER i

* initialize
	DO i = 1,n
	   indices(i) = i
 	ENDDO

* replace indices that are out of range with missing value flags
	DO i = 1, n
	  IF (list(i) .GT. valid_max
     .   .OR. list(i) .LT. valid_min ) list(i) = bad_flag_in
 	ENDDO

* sort
	CALL HEAP2(list, indices, bad_flag_in, n)

* count valid points
	nvalid = n
	DO i = 1,n
	   IF ( list(i) .EQ. bad_flag_in ) nvalid = nvalid - 1
 	ENDDO

* reconcile bad flags and count valid points
	IF ( bad_flag_in .NE. bad_flag_out ) THEN
	   DO i = 1,n
	      IF ( list(i).EQ.bad_flag_in ) list(i) = bad_flag_out
 	   ENDDO
	ENDIF

	RETURN
	END
