	SUBROUTINE COUNT_NUMBER_LIST( string, n )

* count the number of values in a list of the form "{ 1, 3,5,7, 9}"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: *sh* 12/98
* V510: *sh* 10/99 bug fix: single character {3} gave n=0
* V530:  8/00 *sh* - added support for arrays of strings

* calling argument declarations
	INTEGER		n
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		in_quote, no_chars_yet
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
	   IF (in_quote) THEN

#ifdef NO_DOUBLE_ESCAPE_SLASH
              IF (c .EQ. '\') THEN
#else
              IF (c .EQ. '\\') THEN
#endif
	         i = i + 1   ! skip processing of next character 
	      ELSEIF  (c .EQ. '"') THEN
	         in_quote = .FALSE.
	      ELSE
	         CONTINUE   ! skip over anything inside of quotations
	      ENDIF
	   ELSEIF (c .EQ. '"') THEN
	      in_quote = .TRUE.
	   ELSEIF (c .EQ. ',') THEN
	      n = n+1
	   ELSEIF (c.EQ.'s' .OR. c.EQ.'S' ) THEN
	      IF ( i+7 .GT. slen ) CONTINUE  ! trap syntax error later
	      vax_code = STR_CASE_BLIND_COMPARE(string(i:i+6), 'SYSTEM:')
	      IF( vax_code .NE. vms_str_success ) CONTINUE ! trap syntax later
	      i = i + 6
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
