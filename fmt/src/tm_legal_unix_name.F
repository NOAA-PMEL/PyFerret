	LOGICAL FUNCTION TM_LEGAL_UNIX_NAME( name )
*
* determine whether a particular string is a "legal" Unix path name
*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 1/93 - based on tm_legal_name
*
* 2jun94 *kob* - first statement function too long  - caught by ibm compiler
* calling argument declarations:
	CHARACTER*(*) name

* internal variable declarations:
	INTEGER	TM_LENSTR1, slen, i
	LOGICAL	LETTER, DIGIT, PUNCT
	CHARACTER c*1

* statement functions
	LETTER( c ) = (c.GE.'A' .AND. c.LE.'Z') .OR. 
     .                    (c.GE.'a' .AND. c.LE.'z')
	DIGIT ( c ) = c.GE.'0' .AND. c.LE.'9'
	PUNCT ( c ) = c.EQ.'.' .OR. c.EQ.'_' .OR.  c.EQ.'/'

* string length
	slen = TM_LENSTR1( name )

* blank name is not allowed
	IF ( name .EQ. ' ' ) GOTO 5000

* all characters must be letters, digits or punctuation
	DO 100 i = 2, slen
	   c = name(i:i)
	   IF ( .NOT.(LETTER(c) .OR. DIGIT(c) .OR. PUNCT(c)) ) GOTO 5000
 100	CONTINUE

* it's ok
	TM_LEGAL_UNIX_NAME = .TRUE.
	RETURN

* it's bad
 5000	TM_LEGAL_UNIX_NAME = .FALSE.
	RETURN

	END
