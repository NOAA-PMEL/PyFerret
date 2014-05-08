	SUBROUTINE PARSE_NUMBER_LIST( string, array, nmax, n, status )

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
* return an array of floating point values from a list of the form
* "{ 1, 3,5,7, 9}"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: *sh* 12/98
*	   2/99 - bug fix - 2 commas together
*       *acm* 3/12 cleanup ifdefs and unnecessary include files. Add disclaimer.
* V69+  *acm* 5/14 ticket 2174: check that the expression is just a constant
*                  before reading from the buffer into the array.

* calling argument declarations
	INTEGER		nmax, n, status
	REAL		array(nmax)
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		TM_NUMBER
	INTEGER 	TM_LENSTR1, comma_pos, last_comma_pos, slen
	CHARACTER	buff*64

	include 'ferret.parm'
	include 'errmsg.parm'

* initialize
	n = 0
	slen	  = LEN( string )
	last_comma_pos = 0

* trim around the enclosing "{}" brackets
	IF ( string(slen:slen) .EQ. '}' ) slen = slen - 1
	IF ( string (1:1) .EQ. '{' ) last_comma_pos = 1
	IF (last_comma_pos .GE. slen) GOTO 2000

* loop through the comma list decoding and returning values
 100	CONTINUE
	IF (last_comma_pos .GE. slen) THEN
	   comma_pos = 0	! last item
	ELSE
	   comma_pos = INDEX( string(last_comma_pos+1:slen), ',' )
	ENDIF
	n = n + 1
	If ( n .GT. nmax ) GOTO 5100   ! should never happen
	IF ( comma_pos .EQ. 0 ) THEN
*  ... last item in list
	   IF (last_comma_pos .GE. slen) THEN
	       array(n) = bad_val4			! comma as last item
	   ELSE
	      buff = string(last_comma_pos+1:slen)
	      IF ( buff .EQ. ' ' ) THEN
	         array(n) = bad_val4
	      ELSE
	         IF (TM_NUMBER (buff)) THEN
	            READ ( buff, *, ERR=5000 ) array(n)
	         ELSE
		    GOTO 5200
	         ENDIF
	      ENDIF
	   ENDIF
	   GOTO 1000
	ELSEIF ( comma_pos .EQ. 1 ) THEN
	   array(n) = bad_val4				! 2 commas together
	ELSE
* ... read another value
	   buff = string(last_comma_pos+1:last_comma_pos+comma_pos-1)
	   IF ( buff .EQ. ' ' ) THEN
	      array(n) = bad_val4
	   ELSE
	      IF (TM_NUMBER (buff)) THEN
	         READ ( buff, *, ERR=5000 ) array(n)
	      ELSE
	         GOTO 5200
	      ENDIF
	   ENDIF
	ENDIF
	last_comma_pos = last_comma_pos + comma_pos
	GOTO 100

* success
 1000	status = ferr_ok
	RETURN

* null brackets
 2000	n = 1
	array(1) = bad_val4
	GOTO 1000

* syntax error
 4999	RETURN
 5000	CALL ERRMSG( ferr_syntax, status, string(last_comma_pos+1:),
     .							*4999 )
 5100	CALL ERRMSG( ferr_internal, status, string, *4999 )
 5200   CALL ERRMSG( ferr_syntax, status, 
     .          'must be a constant '//
     .		buff(:TM_LENSTR1(buff)), *4999 )

	END
