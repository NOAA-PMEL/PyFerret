	LOGICAL FUNCTION TM_DIGIT( string )

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
* test to see if the first character of "string" is the first character
* of a number

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 06/24/88
* 11/6/94 *sh* Kludge for AIX port (fixes to PLOT+ to avoid problems with
*	FORTRAN READ ERR= branches):  check for "/" following "." since
*	path names often begin with "./name"
* 1/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an character constant instead
* V685+ 12/13    - *acm*  bounds checking: dont refer to parts of the string outside its bounds.

#include "gt_lib.parm"

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		i, slen
	CHARACTER*1	c
	
* local parameter declarations:
        CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

	slen = LEN(string)
	IF (slen .EQ. 0) THEN
	   TM_DIGIT = .FALSE.
	   RETURN
	ENDIF

* skip leading blanks
	DO 100 i = 1, slen
	   c = string(i:i)
	   IF ( c .NE. ' ' .AND. c .NE. tab ) GOTO 200
 100	CONTINUE

* string is blank
	TM_DIGIT = .FALSE.
        RETURN
	
* check first character
 200	TM_DIGIT =   c .EQ. '+'
     .		.OR. c .EQ. '-'
     .		.OR. c .EQ. '.'
     .		.OR. c .GE. '0' .AND. c .LE. '9'

* kludge check for "./name" (11/94 for AIX port)
	IF (slen.GE.2 ) THEN
	   IF (c.EQ.'.') TM_DIGIT = string(2:2) .NE. PathSeparator

* Check for .. starting a string, dataset global attributes

           IF (i .LT. slen) THEN
              IF(c .EQ. '.' .AND. string(i+1:i+1) .EQ. '.') TM_DIGIT = .FALSE.
	   ENDIF
	
	ENDIF

	RETURN

      END
