	CHARACTER*(*) FUNCTION COMPOUND_KEY( cx_list, ncx, flags, tlen )

* create a full title for a plot 
* - possibly with more than one variable documented

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   1/4/89

* calling argument declarations:
	INTEGER	ncx, cx_list(ncx), flags(0:10), tlen

* internal variable declarations:
	CHARACTER*200	KEY_STRING
	INTEGER		slen, maxlen, i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* maximum title length
	maxlen = LEN( COMPOUND_KEY )

* initialize to the first variable context
	COMPOUND_KEY = KEY_STRING( cx_list(1), flags, tlen )

* add any remaining variables separated by commas
	DO 100 i = 2, ncx
	   COMPOUND_KEY = COMPOUND_KEY(:tlen)//' , '//
     .			  KEY_STRING( cx_list(i), flags, slen )
	   tlen = MIN( maxlen, tlen+slen+3 )
 100	CONTINUE

* in case it was too long give an indication
	IF ( tlen .EQ. maxlen ) COMPOUND_KEY(tlen:tlen) = '*'

	RETURN
	END
