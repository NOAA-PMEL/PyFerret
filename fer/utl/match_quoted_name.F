	LOGICAL FUNCTION MATCH_QUOTED_NAME ( test_string, model_string)

* check to see if test string is same as model string to full length of model
* string
* model string must be all upper case - test string ignores case
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/1/86
* revision 0.1 - 12/9/86 - allowed model_string of arbitrary length
* V495: 7/7/98 - *sh* if test name is in single quotes use case-sensitive comp.

* calling argument declarations:
	CHARACTER*(*)	model_string
	CHARACTER*(*)	test_string

* internal variable declarations:
	LOGICAL         is_quoted
	INTEGER		TM_LENSTR, len_model, len_test, len_model_chars, i
	CHARACTER*1	UPPER_CASE

	len_model	= LEN( model_string )
	len_model_chars	= TM_LENSTR( model_string )
	len_test	= TM_LENSTR( test_string(1:LEN(test_string)) )

* (7/98) is test string in quotations?
	is_quoted = test_string(1:1) .EQ. '\'' 
	IF (is_quoted) is_quoted = test_string(len_test:len_test) .EQ. '\''
	IF (is_quoted) len_test = len_test - 2

* test string cannot be shorter than model or longer than fully terminated model
	IF (   len_test        .LT. len_model_chars
     .	 .OR.( len_model_chars .LT. len_model		! fully terminated ?
     .	 .AND. len_test	       .GT. len_model_chars )	) GOTO 200


	IF (is_quoted) THEN
	   DO 100 i = 1,len_model_chars
	      IF (model_string(i:i) .NE. test_string(i+1:i+1)) GOTO 200
 100	   CONTINUE
	ELSE
	   DO 110 i = 1,len_model_chars
	      IF ( model_string(i:i) .NE. 
	1	   UPPER_CASE( test_string(i:i) ) ) GOTO 200
 110	   CONTINUE
	ENDIF

* it matches
	MATCH_QUOTED_NAME = .TRUE.
	RETURN

* it doesn't match
 200	MATCH_QUOTED_NAME = .FALSE.
	RETURN

	END
