	LOGICAL FUNCTION MATCH_TEMPLATE (name, template)

* check to see if the name matches the template
* the template may contain
*	"*" which matches 0 or more of any characters 
*	"?" which match any single character

* Note: this routine IS case sensitive so make sure to UPCASE strings as
*	needed prior to calling this

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420  4/18/96

* calling argument declarations:
	CHARACTER*(*)	template
	CHARACTER*(*)	name

* internal variable declarations:
	LOGICAL		star_skip
	INTEGER		TM_LENSTR1, in, it, nlen, tlen
	CHARACTER*1	ch

* initialize
	nlen	= TM_LENSTR1( name )
	tlen	= TM_LENSTR1( template )
	star_skip = .FALSE.
	in = 1

* LOOP through the characters of the template
* (advance the char ptr for name, "in", as needed)
	DO 500 it = 1, tlen

	  ch = template(it:it)		! next character to match

	  IF ( ch .EQ. '*' ) THEN
	    star_skip = .TRUE.
	    GOTO 500

	  ELSEIF ( ch .EQ. '?' ) THEN
	    IF ( star_skip ) THEN
	      GOTO 500			! "*?" matches same as "*"
	    ELSE
	      GOTO 400
	    ENDIF

	  ELSEIF ( star_skip ) THEN
	    DO 100 in = in, nlen
	      IF ( name(in:in) .EQ. ch ) THEN
	        star_skip = .FALSE.
	        GOTO 400
	      ENDIF
 100	    CONTINUE
	    GOTO 1000			! no match for char after "*"

	  ELSEIF ( in .GT. nlen ) THEN
	    GOTO 1000			! no more name chars available

	  ELSEIF ( ch .EQ. name(in:in) ) THEN
	    GOTO 400

	  ELSE
	    GOTO 1000

	  ENDIF

* advance to next character of name
 400	  IF ( (it.EQ.tlen) ) THEN
	    IF ( (in.NE.nlen) ) GOTO 1000  ! too many chars in name
	  ENDIF
	  in = in + 1

 500	CONTINUE

* success - we've passed through the entire template with no error
	MATCH_TEMPLATE = .TRUE.
	RETURN

* something doesn't match
 1000	MATCH_TEMPLATE = .FALSE.
	RETURN

	END
