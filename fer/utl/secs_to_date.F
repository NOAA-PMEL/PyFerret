	CHARACTER*(*) FUNCTION SECS_TO_DATE ( num_secs )
* Convert secs since 01-JAN-0000 00:00:00 to a date of the form
* $_-mm-dd hh:mm:ss
* difference from TM_SECS_TO_DATE: if year=0 or 1 then blank year field 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - code pulled from TM_SECS_TO_DATE
* revision 1.0 - 2/22/89 - calls TM_SECS_TO_DATE

* Calling argument declarations:
	REAL*8		num_secs

* internal variable declarations
	INTEGER		year
	CHARACTER	TM_SECS_TO_DATE*20, temp_str*20

* convert to date string always including year
	temp_str = TM_SECS_TO_DATE( num_secs )

* determine year from dd-mmm-$_:hh:mm:ss
	READ ( temp_str, '(7X,I4)', ERR=5000 ) year

* blank out year if 0000 or 0001
	IF ( year .LE. 1 ) temp_str(7:11) = ' '

* done
	SECS_TO_DATE = temp_str
	RETURN

* internal error
 5000	STOP 'SECS_TO_DATE'
	END
