	SUBROUTINE EQUAL_STRING( string, val, status )

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
* from a string of the form "name=value", return the "value" string
* if the string is simply "name" then return ' ' (blank)
* change the string to upper case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/27/88 - based on EQUAL_VAL
* V200:   7/7/89 -  remove leading blanks, process quotation marks
*	  9/22/89 - fixed bug for NAME="" (close quotes)
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V693+ *sh* 2/15 - added support for _DQ_ quotations
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string, val

* internal variable declarations
	INTEGER	TM_LENSTR1, STR_UPCASE, equal_pos, slen, strt, vax_code

	include 'ferret.parm'
	include 'errmsg.parm'

* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   val = ' '				! "name", alone

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSE

* find first non-blank
	   DO strt = equal_pos+1, slen
	      IF ( string(strt:strt) .NE. ' ' ) GOTO 110
 	   ENDDO
	   STOP '=_str'		! cannot occur

* remove quotation marks
 110	   IF ( string(strt:strt) .EQ. '"'
     .	  .AND. string(slen:slen) .EQ. '"' ) THEN
	      strt = strt + 1
	      slen = slen - 1
	   ELSEIF (string(strt:strt) .EQ. '_'
     .	     .AND. string(slen:slen) .EQ. '_' 
     .	     .AND. slen-strt.GE.8            ) THEN
	      IF (string(strt:strt+3) .EQ. '_DQ_'
     .	    .AND. string(slen-3:slen) .EQ. '_DQ_' ) THEN
	         strt = strt + 4
	         slen = slen - 4
	      ENDIF
	   ENDIF

	   IF ( strt .GT. slen ) THEN
	      val = ' '
	   ELSE
	     vax_code = STR_UPCASE( val, string(strt:slen) )
	   ENDIF

	ENDIF

* success
	status = ferr_ok
 1000	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
	END
