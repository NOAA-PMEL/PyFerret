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
* if true val1=bad_flag if all points are bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/88
* revision 0.1 - 9/22/88 - bug fix.  start at subscript 1

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	npts
	REAL 	array( npts ), bad_flag, val1

* internal variable declarations:
	INTEGER	i

* comparison value
	val1 = bad_flag

* check all data within the indicated context
	DO 100 i = 1, npts
	   IF ( array( i ) .NE. val1 ) THEN
	      IF     ( val1       .EQ. bad_flag ) THEN
	         val1 = array( i )
	      ELSEIF ( array( i ) .NE. bad_flag ) THEN
	         NO_LINE_RANGE = .FALSE.
	         RETURN
	      ENDIF
	   ENDIF
 100	CONTINUE

* they were all identical --> val1 contains bad_flag if they were all bad
	NO_LINE_RANGE = .TRUE.

	RETURN
	END
