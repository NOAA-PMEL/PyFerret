	LOGICAL FUNCTION TM_LEGAL_UNIX_NAME( name )
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
* determine whether a particular string is a "legal" Unix path name
*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 1/93 - based on tm_legal_name
*
* 2jun94 *kob* - first statement function too long  - caught by ibm compiler
* calling argument declarations:
	CHARACTER*(*) name

* internal variable declarations:
	INTEGER	TM_LENSTR1, slen, i
	LOGICAL	LETTER, DIGIT, PUNCT
	CHARACTER c*1

* statement functions
	LETTER( c ) = (c.GE.'A' .AND. c.LE.'Z') .OR. 
     .                    (c.GE.'a' .AND. c.LE.'z')
	DIGIT ( c ) = c.GE.'0' .AND. c.LE.'9'
	PUNCT ( c ) = c.EQ.'.' .OR. c.EQ.'_' .OR.  c.EQ.'/'

* string length
	slen = TM_LENSTR1( name )

* blank name is not allowed
	IF ( name .EQ. ' ' ) GOTO 5000

* all characters must be letters, digits or punctuation
	DO 100 i = 2, slen
	   c = name(i:i)
	   IF ( .NOT.(LETTER(c) .OR. DIGIT(c) .OR. PUNCT(c)) ) GOTO 5000
 100	CONTINUE

* it's ok
	TM_LEGAL_UNIX_NAME = .TRUE.
	RETURN

* it's bad
 5000	TM_LEGAL_UNIX_NAME = .FALSE.
	RETURN

	END
