	LOGICAL FUNCTION TM_DATE( string )

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
* naive test to see if the given string may be a date
* complete date format is dd-mmm-yy:hh:mm:ss where mmm is a string
* naive test insists that
*     1) first character is a digit
*     2) contains : or -
*     3) if "-" must be followed by 3 characters

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 3/12/91 needed for Unix/RISC port
*
* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		i, ip, slen
	CHARACTER*1	c
	
* local parameter declarations:
        CHARACTER*1     tab
        PARAMETER     ( tab = CHAR(9) )

* initialize assuming failure
	TM_DATE = .FALSE.

* skip leading blanks
        slen = LEN(string)
	DO 100 i = 1, slen
	   c = string(i:i)
	   IF ( c .NE. ' ' .AND. c .NE. tab ) GOTO 200
 100	CONTINUE
* string is blank
	RETURN

* is first character a digit ?
* modified 6/92 -- kob
* 200    IF ( c .LE. '0' .OR. c .GE. '9' ) RETURN
 200    IF ( c .LT. '0' .OR. c .GT. '9' ) RETURN

* is there a colon ?
        IF ( INDEX( string, ':') .GT. 0 ) THEN
           TM_DATE = .TRUE.
           RETURN
        ENDIF

* is there a dash ?
        ip = INDEX( string, '-')
        IF ( ip .EQ. 0 ) RETURN

* do three letters follow the dash ?
        IF ( slen .LT. ip+3 ) RETURN
        DO 300 i = ip+1, ip+3
           c = string(i:i)
           IF ( ( c.GE.'A' .AND. c.LE.'Z' )
     .    .OR.  ( c.GE.'a' .AND. c.LE.'z' ) ) GOTO 300
* not a letter
	   RETURN
 300    CONTINUE

* it meets the requirements
        TM_DATE = .TRUE.
        RETURN

        END
