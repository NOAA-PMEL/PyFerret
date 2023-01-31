	SUBROUTINE EQUAL_VAL( string, val, status )

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
* from a string of the form "name=value", return the value
* if the string is simply "name" then return unspecified value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/14/88
* revision 0.1 -  9/8/88 - explicit check for digit so words beginning with T
*			   aren't treated as .TRUE. by VAX Fortran 
* V420: 10/95 - need to buffer "READ(*)" for AIX bort
*       *acm* 3/12 cleanup ifdefs and unnecessary include files

        IMPLICIT NONE
* calling argument declarations
	INTEGER		status
	REAL		val
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		TM_DIGIT
	INTEGER 	equal_pos, slen

	include 'ferret.parm'
	include 'errmsg.parm'

* initialize
	slen	  = LEN( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   val = unspecified_val4		! "name", alone

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSEIF ( string(equal_pos+1:) .EQ. ' ' ) THEN
	   GOTO 5000				! "name =    "

	ELSEIF ( .NOT.TM_DIGIT(string(equal_pos+1:)) ) THEN
	   GOTO 5000				! "name = text"

	ELSE
	   READ ( string(equal_pos+1:), *, ERR=5000 ) val
	ENDIF

* success
	status = ferr_ok
 1000	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
	END
