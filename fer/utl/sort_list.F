	SUBROUTINE SORT_LIST(n, bad_flag_in, bad_flag_out, list, indices)

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
* Sort an input list of values returning both the sorted list and the
* map vector (as floats)
* if there are missing values in the index list, sort em as if they are BIG


* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510 *sh* 8/99

#ifdef unix
	include 'errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER n
	REAL bad_flag_in, bad_flag_out, list(n), indices(n)

* internal variable declarations:
	LOGICAL jossled
	INTEGER i, j
	REAL tmp

* initialize
	DO 10 i = 1,n
 10	indices(i) = i

* primitive bubble sort (could be replaced)
* sort into ascending order -- "j" is always the lower index
	DO 200 j = 1, n-1
	  jossled = .FALSE.
	  DO 100 i = 2, n-(j-1)
	    IF (list(i-1).EQ.bad_flag_in 
     .     .OR.(list(i).NE.bad_flag_in.AND.list(i-1).GT.list(i)) ) THEN
* ... intercahange order in both output arrays
	      tmp = list(i-1)
	      list(i-1) = list(i)
	      list(i) = tmp
	      tmp = indices(i-1)
	      indices(i-1) = indices(i)
	      indices(i) = tmp
	      jossled = .TRUE.
	    ENDIF
 100	  CONTINUE
	  IF (.NOT.jossled) RETURN
 200	CONTINUE

* reconcile bad flags
	IF ( bad_flag_in .NE. bad_flag_out ) THEN
	   DO 500 i = 1,n
 500	   IF ( list(i).EQ.bad_flag_in ) list(i) = bad_flag_out
	ENDIF

	RETURN
	END
