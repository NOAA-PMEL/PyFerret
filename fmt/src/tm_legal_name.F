	LOGICAL FUNCTION TM_LEGAL_NAME( name )
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
*
* determine whether a particular string is a "legal" variable name
*
* programmer - steve hankin (mods by M. Verschell)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  07/07/89
* TM_:	 12/06/89
* IBM port 14apr94  *kob* : "Letter(c)=...." line to long for IBM compiler
* 6/95 *sh* - allow "$" as a legal symbol since it is used by PLOT+ symbols

* calling argument declarations:
	CHARACTER*(*) name
*
* internal variable declarations:
	INTEGER	TM_LENSTR1, slen, i
	LOGICAL	LETTER, DIGIT
	CHARACTER c*1
*
* statement functions
	LETTER( c ) = (c.GE.'A' .AND. c.LE.'Z') .OR. 
     .                      (c.GE.'a' .AND. c.LE.'z')
	DIGIT ( c ) = c.GE.'0' .AND. c.LE.'9'
*
* string length
	slen = TM_LENSTR1( name )
*
* first character must be a letter (also catches 0 length strings)
	IF ( .NOT.LETTER(name(1:1)) ) GOTO 5000
*
* all other characters must be letters, digits, "_", or "$"
	DO 100 i = 2, slen
	   c = name(i:i)
	   IF ( .NOT.(LETTER(c)
     .	         .OR. DIGIT(c)
     .	         .OR. c.EQ.'_'
     .	         .OR. c.EQ.'$')    ) GOTO 5000
 100	CONTINUE
*
* it's ok
	TM_LEGAL_NAME = .TRUE.
	RETURN
*
* it's bad
 5000	TM_LEGAL_NAME = .FALSE.
	RETURN

	END
