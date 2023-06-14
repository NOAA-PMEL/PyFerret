	LOGICAL FUNCTION TM_DIGIT_TEST( string )

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
* test to see if the string can be read as a number

* Ansley Manke 10/11/05
* from tm_digit.F


#include "gt_lib.parm"

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
        LOGICAl         maybe
	INTEGER		TM_LENSTR1, i, j
	CHARACTER*1	c
	
* local parameter declarations:
        CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

* skip leading blanks
	DO 100 i = 1, LEN(string)
	   c = string(i:i)
	   IF ( c .NE. ' ' .AND. c .NE. tab ) GOTO 200
 100	CONTINUE

* string is blank
	TM_DIGIT_TEST = .FALSE.
	
* check first character
 200	TM_DIGIT_TEST =   c .EQ. '+'
     .		.OR. c .EQ. '-'
     .		.OR. c .EQ. '.'
     .		.OR. c .GE. '0' .AND. c .LE. '9'

* kludge check for "./name" (11/94 for AIX port)
	IF (c.EQ.'.' .AND. LEN(string).GE.2 )
     .		TM_DIGIT_TEST = string(2:2) .NE. PathSeparator

! dataset attributes ..attname
        IF (INDEX(string, '..') .GT. 0) TM_DIGIT_TEST = .FALSE.  

        IF (.NOT. TM_DIGIT_TEST) RETURN

* Check further characters.  There may be an E for exponent or a D for a double precision 

        IF (TM_LENSTR1(string) .LE. i) RETURN
        maybe = .FALSE.
	DO 300 j = i+1, TM_LENSTR1(string)
	   c = string(j:j)
           TM_DIGIT_TEST =   c .EQ. '+'
     .		.OR. c .EQ. '-'
     .		.OR. c .EQ. '.'
     .		.OR. c .GE. '0' .AND. c .LE. '9'
     .		.OR. c .EQ. 'E'
     .		.OR. c .EQ. 'e'
     .		.OR. c .EQ. 'D'
     .		.OR. c .EQ. 'd'
           IF (.NOT. TM_DIGIT_TEST) RETURN

           IF (maybe) THEN  ! previously hit an E or D, see if this next char is numeric
              TM_DIGIT_TEST =   c .EQ. '+'
     .		.OR. c .EQ. '-'
     .		.OR. c .EQ. '.'
     .		.OR. c .GE. '0' .AND. c .LE. '9'
              maybe = .FALSE.
              IF (.NOT. TM_DIGIT_TEST) RETURN
           ENDIF

           maybe = c .EQ. 'E'
     .		.OR. c .EQ. 'e'
     .		.OR. c .EQ. 'D'
     .		.OR. c .EQ. 'd'
 300	CONTINUE
        IF (maybe) TM_DIGIT_TEST = .FALSE.   ! ended with D or E

	RETURN

      END
