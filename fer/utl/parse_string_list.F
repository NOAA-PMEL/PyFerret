	SUBROUTINE PARSE_STRING_LIST( cx, string, mr, status )

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
