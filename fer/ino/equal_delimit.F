	SUBROUTINE EQUAL_DELIMIT( string, outstring, status )

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
* from a string of the form "DELIMITER=xxxx" determine the list of field
* delimiters and return in outstring

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530: *sh* 11/00
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string, outstring

* internal variable declarations
	INTEGER	TM_LENSTR1, n, llen, pos, ll, ic
	CHARACTER buff*80, buff1*8

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'

* internal parameter declarations:
        CHARACTER*1     tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

* input is comma separated, may include "\t", "\,", and HEX characters

* get the input string
        CALL EQUAL_STR_LC( string, buff, status )   ! lower case 11/92
        IF ( status .NE. ferr_ok ) RETURN

! initialize
	n = 0
	llen = TM_LENSTR1(buff)
	
* parse out the next field (implied DO-WHILE loop)
 100	pos = INDEX(buff,',')   ! points to the terminating comma
	  IF (pos.EQ.0) THEN
	    pos = llen + 1
	  ELSEIF (pos.EQ.1) THEN
	    GOTO 5100   ! null delimiter
*  ... escaped comma?
	  ELSEIF (buff(pos-1:pos-1).EQ.'\') THEN
	    ll = INDEX(buff(pos+1:),',')
	    IF (ll.EQ.0) THEN
	      pos = llen + 1
	    ELSE
	      pos = pos + ll
	    ENDIF 
	  ENDIF

* interpret the next delimiter
	  n = n + 1
	  IF (buff(1:1).EQ.'\') THEN
	    IF (buff(2:2).EQ.'t' .OR. buff(2:2).EQ.'T') THEN
	      outstring(n:n) = tab
	    ELSEIF (buff(2:2).EQ.'b' .OR. buff(2:2).EQ.'B' ) THEN
	      outstring(n:n) = ' '
	    ELSEIF (buff(2:2).EQ.'n' .OR. buff(2:2).EQ.'N') THEN
	      outstring(n:n) = CHAR(12) ! newline character
	    ELSE
	      outstring(n:n) = buff(2:2)
	    ENDIF
	  ELSEIF ( buff(1:1).GE.'0'
     .	     .AND. buff(1:1).LE.'9'
     .	     .AND. pos.GT.2 ) THEN
	    buff1 = buff(:pos-1)
	    READ(buff1,'(I4)',ERR=5100) ic
	    IF (ic.GT.127) GOTO 5100
	    outstring(n:n) = CHAR(ic)  ! any decimal value
	  ELSEIF (pos.GT.2) THEN
	     GOTO 5100
	  ELSE
	    outstring(n:n) = buff(1:1)
	  ENDIF

* ... shift to the next field in the buffer
	  buff = buff(pos+1:)
	  llen = llen - pos
	IF (llen.GT.0) GOTO 100 
* -- end of implied DO-WHILE loop

* terminate with a NULL (C-style)
	outstring(n+1:n+1) = CHAR(0)

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, buff(pos:llen), *5000 )
	END
