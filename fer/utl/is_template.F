	LOGICAL FUNCTION IS_TEMPLATE (string)

* check to see if the string is a template - i.e. it contains "*" or "?"
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420  4/18/96

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
* ...	

	IS_TEMPLATE = INDEX(string,'*') .GT. 0
     .		 .OR. INDEX(string,'?') .GT. 0
	RETURN

	END
