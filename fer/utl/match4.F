	LOGICAL FUNCTION MATCH4 ( test_string, model_string)

* check to see if test string is same as model string for as many characters
* as it is given (to a max of 4)
* model string must be all upper case - test string ignores case
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/24/86

	INTEGER		TM_LENSTR, len_model, len_test, len_to_test, i

	CHARACTER*1	UPPER_CASE
	CHARACTER*4	model_string
	CHARACTER*(*)	test_string

	len_model = TM_LENSTR( model_string )
	len_test  = TM_LENSTR( test_string(1:LEN(test_string)) )

* if model string is fewer than 4 characters the test string must not be longer
	IF (      len_model .LT. 4
     .	    .AND. len_test  .GT. len_model ) GOTO 200

	len_to_test = MIN( len_test, len_model )

	DO 100 i = 1,len_to_test
	IF ( model_string(i:i) .NE. UPPER_CASE( test_string(i:i) ) ) GOTO 200
 100	CONTINUE

* it matches
	MATCH4 = .TRUE.
	RETURN

* it doesn't match
 200	MATCH4 = .FALSE.
	RETURN

	END
