	CHARACTER*(*) FUNCTION COMPOUND_TITLE( cx_list, ncx, tlen )

* create a full title for a plot 
* - possibly with more than one variable documented

* support routines:
*	VAR_TITLE	   - gets gets a simple title string for variable
*			     eg. "TEMPERATURE"
*	VAR_UNITS	   - gets a string with the units for this variable
*	VAR_TITLE_MOD	   - gets a modification to the title string, if any
*			     eg. "( averaged in X )"
*	FULL_VAR_TITLE	   - VAR_TITLE//VAR_UNITS//VAR_TITLE_MOD with parens
*			     eg. "TEMPERATURE ( averaged in X )"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   1/4/89 - from COMPOUND_PLOT_TITLE: major re-write without colors

* calling argument declarations:
	INTEGER	ncx, cx_list(ncx), tlen

* internal variable declarations:
	CHARACTER*200	FULL_VAR_TITLE
	INTEGER		slen, maxlen, i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* maximum title length
	maxlen = LEN( COMPOUND_TITLE )

* initialize to the first variable context
	COMPOUND_TITLE = FULL_VAR_TITLE( cx_list(1), tlen )

* add any remaining variables separated by commas
	DO 100 i = 2, ncx
	   COMPOUND_TITLE = COMPOUND_TITLE(:tlen)//' , '//
     .			    FULL_VAR_TITLE( cx_list(i), slen )
	   tlen = MIN( maxlen, tlen+slen+3 )
 100	CONTINUE

* in case it was too long give an indication
	IF ( tlen .EQ. maxlen ) COMPOUND_TITLE(tlen:tlen) = '*'

	RETURN
	END
