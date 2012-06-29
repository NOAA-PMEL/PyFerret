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

* calling argument declarations
	INTEGER		cx, mr, status
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		stored, in_quote, doub_quote, do_system_cmnd
	INTEGER 	TM_LENSTR1, STR_CASE_BLIND_COMPARE,
     .			i, n, nnew, spos, slen, str_start, vax_code, istat
	CHARACTER	c*1

* internal parameter declarations
	CHARACTER*1     tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

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
* (poor man's FORTRAN DO-WHILE loop)
	i = spos - 1
 100	i = i + 1
	   c = string(i:i)
	   IF (in_quote) THEN
#ifdef NO_DOUBLE_ESCAPE_SLASH
              IF (c .EQ. '\') THEN
#else
              IF (c .EQ. '\\') THEN
#endif
	         i = i + 1   ! skip processing of next character 
	      ELSEIF  (c .EQ. '"' .AND. doub_quote
     .		 .OR.  c .EQ. "'" .AND. .NOT.doub_quote ) THEN
* ... end of quotated string -- save the string
	         in_quote = .FALSE.
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
	      ELSE
	         CONTINUE   ! skip over anything inside of quotations
	      ENDIF
	   ELSEIF (c.EQ.'"' .OR. c.EQ."'") THEN
	      IF (stored) GOTO 5000
	      in_quote = .TRUE.
	      doub_quote = c.EQ.'"'
	      str_start = i + 1
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
	IF ( i .LT. slen ) GOTO 100
	IF (in_quote) GOTO 5000

* success
 1000	status = ferr_ok
	RETURN

* syntax error
 4999	RETURN
 5000	CALL ERRMSG( ferr_syntax, status, string(i:), *4999 )

	END
