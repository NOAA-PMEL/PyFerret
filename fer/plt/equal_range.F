	SUBROUTINE EQUAL_RANGE( string,idim,lo,hi,delta,formatted,
     .				range_rqd, cal_id, status )

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
* from a string of the form "name=value", return the lo/high value range
* if the string is simply "name" then return unspecified values
* the "delta" value is optional - return unspecfied_val8 if not given

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V3.00  20/10/93 - *sh*
* V3.20  18/08/94 - *kob*  IBM port requires use of RISC buff for string concat's
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V500 *sh* - 6/99 check for quotations around text (e.g. date strings)
* V521 *sh* - added boolean "range_rqd" argument
* V530 *acm* - send cal_id to TRANSLATE_LIMIT
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* V693+ *sh* 2/15 - added support for _DQ_ quotations
*		    mixing double quote marks and _DQ_ may lead to a bad place

        IMPLICIT NONE
* calling argument declarations
	LOGICAL		formatted, range_rqd
	INTEGER		idim, cal_id, status
	REAL*8		lo, hi, delta
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		has_delta, ldummy, not_by_ss, in_quote, in_DQ
	PARAMETER 	(not_by_ss =  .FALSE.)
	INTEGER 	equal_pos, colon1, colon2, end2, slen, i
	REAL*8		tmp

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xrisc.cmn'

* initialize
	status = ferr_ok
	slen	  = LEN( string )
	equal_pos = INDEX( string, '=' )

* check for gross syntax errors
	IF ( equal_pos .EQ. 0 ) THEN
	   GOTO 5000
	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="
	ELSEIF ( string(equal_pos+1:) .EQ. ' ' ) THEN
	   GOTO 5000				! "name =    "
! removed 6/99 so quotations wouldn't be trapped here
!	ELSEIF ( .NOT.TM_DIGIT(string(equal_pos+1:)) ) THEN
!	   GOTO 5000				! "name = text"
	ENDIF

* lo:hi:delta parsing and syntax checking
	colon1 = 0
	in_quote = .FALSE.
	in_DQ    = .FALSE.
	i = equal_pos
	DO WHILE (i .LT. slen)
	   i = i + 1
	   IF (string(i:i) .EQ. '"') THEN
	      in_quote = .NOT.in_quote
	   ELSEIF (string(i:i) .EQ. '_' .AND. i+3 .LE. slen ) THEN
	      IF (string(i:i+3) .EQ. "_DQ_" ) THEN
	         in_DQ = .NOT.in_DQ
	         i = i + 3
	      ENDIF
	   ELSEIF (in_quote .OR. in_DQ) THEN
	      CONTINUE	! skip text inside of quotation
	   ELSEIF (string(i:i) .EQ. ':') THEN
	      colon1 = i
	      GOTO 110
	   ENDIF
	ENDDO
	IF ( in_quote .OR. in_DQ) GOTO 5000
	IF (colon1 .EQ. 0 ) GOTO 5000

 110	colon2 = colon1
	i = colon1
	DO WHILE (i .LT. slen)
	   i = i + 1
	   IF (string(i:i) .EQ. '"') THEN
	      in_quote = .NOT.in_quote
	   ELSEIF (string(i:i) .EQ. '_' .AND. i+3 .LE. slen ) THEN
	      IF (string(i:i+3) .EQ. "_DQ_" ) THEN
	         in_DQ = .NOT.in_DQ
	         i = i + 3
	      ENDIF
	   ELSEIF (in_quote .OR. in_DQ) THEN
	      CONTINUE	! skip text inside of quotation
	   ELSEIF (string(i:i) .EQ. ':') THEN
	      colon2 = i
	      GOTO 210
	   ENDIF
	ENDDO
	IF ( in_quote ) GOTO 5000
 210	has_delta = colon2 .GT. colon1
	IF ( has_delta ) THEN
	    end2 = colon2 - 1
	    IF ( slen .LE. colon2 ) GOTO 5000
	ELSE
	   end2 = slen
	ENDIF
	IF ( end2 .LE. colon1 ) GOTO 5000

* translate the low limit
	CALL TRANSLATE_LIMIT( string(equal_pos+1:colon1-1), idim,
     .			      not_by_ss, lo, formatted, cal_id, status )
	IF ( status .NE. ferr_ok ) RETURN

* translate the high limit
	CALL TRANSLATE_LIMIT( string(colon1+1:end2), idim,
     .			      not_by_ss, hi, formatted, cal_id, status )
	IF ( status .NE. ferr_ok ) RETURN

* optionally translate the delta value
	IF ( has_delta ) THEN
	   CALL TRANSLATE_LIMIT( string(colon2+1:slen), idim,
     .			      not_by_ss, delta, ldummy, cal_id, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   delta = unspecified_val8
	ENDIF
	
* date strings are returned negative from TRANSLATE_LIMIT
	IF ( formatted .AND. (idim .EQ. t_dim .OR. idim .EQ. f_dim) ) THEN
	   delta = unspecified_val8	! can't modify time axis with delta
	   has_delta = .FALSE.
	   lo = -1 * lo
	   hi = -1 * hi
	   IF ( lo .GT. hi ) THEN	! PLOT+ cant handle reversed dates
	      tmp = lo
	      lo = hi
	      hi = tmp
	   ENDIF
	ENDIF
      
* two possible notations to indicate reserved limits
* change lo:hi:-delta  to hi:lo:delta
	IF ( has_delta .AND. delta .LT. 0.D0 ) THEN
	   IF ( lo .LT. hi ) THEN
	      tmp = lo
	      lo = hi
	      hi = tmp
	   ENDIF
	   delta = ABS( delta )
	ENDIF

* check for valid range
	IF ( range_rqd ) THEN
	   IF ( lo .EQ. hi ) GOTO 5100
	ENDIF

* success
 1000	RETURN

* errors
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
 5100	risc_buff= string
 	CALL ERRMSG( ferr_invalid_command, status,
     .		'Must give a range: '//risc_buff, *1000 )
	END
