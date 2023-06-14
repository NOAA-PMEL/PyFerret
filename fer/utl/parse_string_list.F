	SUBROUTINE PARSE_STRING_LIST( cx, string, mr, status )

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
* return an array of strings from a list of the form
* { "string 1", "string 2", "string 3"}

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530: *sh*  8/00
* V533: *sh*  7/01 - allow single quoted strings
*		   - save null strings for missing values
* V540:  9/01 *sh* - change "SYSTEM:" to "SPAWN:"
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
* 2/07 *acm* - fix bug 1488; if the last item is a missing string, save that.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* 2/15 *sh* added support for _DQ_ and _SQ_

        IMPLICIT NONE
* calling argument declarations
	INTEGER		cx, mr, status
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		stored, in_quote, do_system_cmnd,
     .			doub_atom,  sing_atom,  DQ_atom,  SQ_atom,
     .			doub_quote, sing_quote, DQ_quote, SQ_quote
	INTEGER 	TM_LENSTR1, STR_CASE_BLIND_COMPARE,
     .			i, n, nnew, spos, slen, str_start, vax_code, istat
	CHARACTER	c*1

* internal parameter declarations
	CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xvariables.cmn'

* initialize
	n      = 1
	spos   = 1
	slen   = TM_LENSTR1( string )
	stored    = .FALSE.
	in_quote  = .FALSE.
	do_system_cmnd = .FALSE.

* trim around the enclosing "{}" brackets
	IF ( string(slen:slen) .EQ. '}' ) slen = slen - 1

* dont process trailing blanks
        i = slen
        DO WHILE ( string(i:i) .EQ. ' ' ) 
           slen = slen - 1
           i = i-1
        ENDDO

	IF ( string (1:1) .EQ. '{' ) spos = 2
	IF ( spos .GE. slen ) GOTO 5000

* loop through the text counting commas
	i = spos - 1
	DO WHILE ( i .LT. slen )
	   i = i + 1 
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
* ... possibly escape the next character
              IF (c .EQ. '\') THEN  
	         i = i + 1   ! skip processing of next character 
	      ELSEIF  (doub_atom .AND. doub_quote
     .	         .OR.  sing_atom .AND. sing_quote 
     .	         .OR.  DQ_atom   .AND. DQ_quote 
     .	         .OR.  SQ_atom   .AND. SQ_quote ) THEN
* ... close of paired-quote string -- save the string
	         IF (str_start .LT. i) THEN
	            IF ( do_system_cmnd ) THEN
	               CALL STORE_SYS_STRINGS(string(str_start:i-1),
     .					       cx, mr, n, nnew, status)
	               n = n + nnew - 1
	            ELSE
	               CALL STORE_STRING(string(str_start:i-1),
     .				         mr, n-1, status)
	            ENDIF
	         ELSE
	            CALL SAVE_C_STRING(0,0,mr_c_pointer(mr),n-1,istat) ! NULL
	         ENDIF
	         IF (status .NE. ferr_ok) GOTO 5000
	         stored = .TRUE.
	         do_system_cmnd = .FALSE.
	         in_quote = .FALSE.
	         IF (DQ_atom .OR. SQ_atom) i = i + 3
	      ENDIF

	   ELSEIF (doub_atom.OR.sing_atom.OR.DQ_atom.OR.SQ_atom) THEN
*  ... opening a paired-quote string
	      doub_quote = doub_atom
	      sing_quote = sing_atom
	      DQ_quote   = DQ_atom
	      SQ_quote   = SQ_atom
	      IF (stored) GOTO 5000
	      in_quote = .TRUE.
	      IF (DQ_quote .OR. SQ_quote ) THEN
	         str_start = i + 4
	      ELSE
	         str_start = i + 1
	      ENDIF
	   ELSEIF (c .EQ. ',') THEN
	      IF (.NOT.stored) THEN
	        CALL SAVE_C_STRING(0,0,mr_c_pointer(mr),n-1,istat) ! NULL
	      ENDIF
	      IF (i.EQ.slen) THEN     ! missing string at the end of the list
	        CALL SAVE_C_STRING(0,0,mr_c_pointer(mr),n,istat) ! NULL
	      ENDIF
	      n = n+1
	      stored = .FALSE.
	      do_system_cmnd = .FALSE.
	   ELSEIF (c.EQ.'s' .OR. c.EQ.'S' ) THEN
	      IF ( i+6 .GT. slen ) GOTO 5000
	      vax_code = STR_CASE_BLIND_COMPARE(string(i:i+5), 'SPAWN:')
	      IF( vax_code .NE. vms_str_success ) GOTO 5000
	      i = i + 5
	      do_system_cmnd = .TRUE.
	   ELSE
	      IF (c .NE. ' '
     .	    .AND. c .NE. tab ) GOTO 5000
	   ENDIF
	ENDDO
	IF (in_quote) GOTO 5000

* success
 1000	status = ferr_ok
	RETURN

* syntax error
 4999	RETURN
 5000	CALL ERRMSG( ferr_syntax, status, string(i:), *4999 )

	END
