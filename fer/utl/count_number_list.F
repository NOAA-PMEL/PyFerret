	SUBROUTINE COUNT_NUMBER_LIST( string, n )


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
* count the number of values in a list of the form "{ 1, 3,5,7, 9}"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: *sh* 12/98
* V510: *sh* 10/99 bug fix: single character {3} gave n=0
* V530:  8/00 *sh* - added support for arrays of strings
* V530:  7/01 *sh* - allow single quoted strings
* V540:  9/01 *sh* - change "SYSTEM:" to "SPAWN:"
*       *acm* 3/12   add disclaimer
* V693+ *sh* 2/15 - add support for _DQ_ and _SQ_ quotations

        IMPLICIT NONE
* calling argument declarations
	INTEGER		n
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		in_quote, no_chars_yet,
     .			doub_quote, sing_quote, DQ_quote, SQ_quote,
     .			doub_atom,  sing_atom,  DQ_atom,  SQ_atom
	INTEGER 	TM_LENSTR1, STR_CASE_BLIND_COMPARE,
     .			i, start, slen, vax_code
	CHARACTER*1	c

!* internal parameter declarations
!	CHARACTER*1     tab
!	PARAMETER     ( tab = CHAR(9) )

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'

* initialize
	n = 1
	slen = TM_LENSTR1( string )
	start = 1
	no_chars_yet = .TRUE.
	doub_quote = .FALSE.
	sing_quote = .FALSE.
	DQ_quote   = .FALSE.
	SQ_quote   = .FALSE.

* trim around the enclosing "{}" brackets
	IF ( string(slen:slen) .EQ. '}' ) slen = slen - 1
	IF ( string (1:1) .EQ. '{' ) start = start + 1
	IF (start .GT. slen) RETURN 

* loop through the text counting commas
* (poor man's FORTRAN DO-WHILE loop)
	in_quote = .FALSE.
	i = start - 1
 100	i = i + 1
	   c = string(i:i)

* ... identify quotation atoms
	   doub_atom = c.EQ.'"'
	   sing_atom = c.EQ."'"
	   IF (i+3 .LE. slen) THEN
	      DQ_atom = string(i:i+3) .EQ. "_DQ_"
	      SQ_atom = string(i:i+3) .EQ. "_SQ_"
	   ELSE
	      DQ_atom = .FALSE.
	      SQ_atom = .FALSE.
	   ENDIF

	   IF (in_quote) THEN

              IF (c .EQ. '\') THEN  
	         i = i + 1   ! skip processing of next character 
	      ELSEIF  (     doub_quote .AND. doub_atom
     .	               .OR. sing_quote .AND. sing_atom 
     .	               .OR. DQ_quote   .AND. DQ_atom 
     .	               .OR. SQ_quote   .AND. SQ_atom ) THEN
	         in_quote   = .FALSE.
	         doub_quote = .FALSE.
	         sing_quote = .FALSE.
	         DQ_quote   = .FALSE.
	         SQ_quote   = .FALSE.
	      ELSE
	         CONTINUE   ! skip over anything inside of quotations
	      ENDIF
	   ELSEIF (doub_atom) THEN
	      in_quote   = .TRUE.
	      doub_quote = .TRUE.
	   ELSEIF (sing_atom) THEN
	      in_quote   = .TRUE.
	      sing_quote = .TRUE.
	   ELSEIF (DQ_atom  ) THEN
	      in_quote   = .TRUE.
	      DQ_quote   = .TRUE.
	   ELSEIF (SQ_atom  ) THEN
	      in_quote   = .TRUE.
	      SQ_quote   = .TRUE.
	   ELSEIF (c .EQ. ',') THEN
	      n = n+1
	   ELSEIF (c.EQ.'s' .OR. c.EQ.'S' ) THEN
	      IF ( i+7 .GT. slen ) CONTINUE  ! trap syntax error later
	      vax_code = STR_CASE_BLIND_COMPARE(string(i:i+5), 'SPAWN:')
	      IF( vax_code .NE. vms_str_success ) CONTINUE ! trap syntax later
	      i = i + 5
	      n = n + 2000 - 1   ! allow default memory for 2000 strings 
	   ELSE
	      CONTINUE   ! ignore the text, itself
	   ENDIF
!	   IF (no_chars_yet) THEN
!	      IF (c .NE. ','
!     .	    .AND. c .NE. ' '
!     .	    .AND. c .NE. tab ) no_chars_yet = .FALSE.
!	   ENDIF
	IF ( i .LT. slen ) GOTO 100

!* if there were any digits at all then the count of numbers is the
!* count of commas plus 1
!	IF (.NOT.no_chars_yet) n = n + 1

* success
	RETURN

	END
