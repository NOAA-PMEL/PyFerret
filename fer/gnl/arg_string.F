	CHARACTER*(*) FUNCTION ARG_STRING( val )

* return a text string representing the value "val"
* if val is unspecified (bad_val4) then return blank, else start with a colon
* if val is integer then format it as such
* declare length of ARG_STRING in calling program to be at least 11

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/21/87
* V230: 8/18/92 - eliminated string(1:) notation for MAC compatibility

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	REAL val

* internal variable declarations:
	INTEGER	  i, llen, arg_string_len
	CHARACTER LEFINT*4

* initialiaze
        arg_string_len = LEN( ARG_STRING )

	IF ( val .EQ. bad_val4 ) THEN
	   ARG_STRING = ' '

	ELSE
	   ARG_STRING(1:1) = ':'
* ... truncate to integer if possible
	   IF ( ABS(val) .LT. 10000 ) THEN
	      i = INT( val )
	   ELSE
	      i = 0
	   ENDIF
* ... use left justified I format if appropriate
	   IF ( FLOAT( i ) .EQ. val ) THEN
	      ARG_STRING(2:arg_string_len) = LEFINT( i, llen )
	   ELSE
	      WRITE ( ARG_STRING(2:arg_string_len), '(1PG10.4)' ) val
	   ENDIF
	ENDIF

	RETURN
	END
