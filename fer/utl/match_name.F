	LOGICAL FUNCTION MATCH_NAME ( test_string, model_string)

* check to see if test string is same as model string to full length of model
* string
* model string must be all upper case - test string ignores case
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/1/86
* revision 0.1 - 12/9/86 - allowed model_string of arbitrary length

* calling argument declarations:
	CHARACTER*(*)	model_string
	CHARACTER*(*)	test_string

* internal variable declarations:
	INTEGER		TM_LENSTR, len_model, len_test, len_model_chars, i
	CHARACTER*1	UPPER_CASE

	len_model	= LEN( model_string )
	len_model_chars	= TM_LENSTR( model_string )
	len_test	= TM_LENSTR( test_string(1:LEN(test_string)) )

* test string cannot be shorter than model or longer than fully terminated model
	IF (   len_test        .LT. len_model_chars
     .	 .OR.( len_model_chars .LT. len_model		! fully terminated ?
     .	 .AND. len_test	       .GT. len_model_chars )	) GOTO 200

	DO 100 i = 1,len_model_chars
	IF ( model_string(i:i) .NE. UPPER_CASE( test_string(i:i) ) ) GOTO 200
 100	CONTINUE

* it matches
	MATCH_NAME = .TRUE.
	RETURN

* it doesn't match
 200	MATCH_NAME = .FALSE.
	RETURN

	END
