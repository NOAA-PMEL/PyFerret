	LOGICAL FUNCTION TM_LEGAL_NAME( name )
*
* determine whether a particular string is a "legal" variable name
*
* programmer - steve hankin (mods by M. Verschell)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  07/07/89
* TM_:	 12/06/89
* IBM port 14apr94  *kob* : "Letter(c)=...." line to long for IBM compiler
* 6/95 *sh* - allow "$" as a legal symbol since it is used by PLOT+ symbols

* calling argument declarations:
	CHARACTER*(*) name
*
* internal variable declarations:
	INTEGER	TM_LENSTR1, slen, i
	LOGICAL	LETTER, DIGIT
	CHARACTER c*1
*
* statement functions
	LETTER( c ) = (c.GE.'A' .AND. c.LE.'Z') .OR. 
     .                      (c.GE.'a' .AND. c.LE.'z')
	DIGIT ( c ) = c.GE.'0' .AND. c.LE.'9'
*
* string length
	slen = TM_LENSTR1( name )
*
* first character must be a letter (also catches 0 length strings)
	IF ( .NOT.LETTER(name(1:1)) ) GOTO 5000
*
* all other characters must be letters, digits, "_", or "$"
	DO 100 i = 2, slen
	   c = name(i:i)
	   IF ( .NOT.(LETTER(c)
     .	         .OR. DIGIT(c)
     .	         .OR. c.EQ.'_'
     .	         .OR. c.EQ.'$')    ) GOTO 5000
 100	CONTINUE
*
* it's ok
	TM_LEGAL_NAME = .TRUE.
	RETURN
*
* it's bad
 5000	TM_LEGAL_NAME = .FALSE.
	RETURN

	END
