	CHARACTER*(*) FUNCTION FULL_VAR_TITLE( cx, tlen )

* create a title for a memory variable including:
* i)   the variable's title, 
* ii)  modifications to the title ( either mods given as a part of the name or
*      remarks about transformations performed on the variable )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/12/87
* revision 0.1 - 8/29/87 - added parend around title mod
* revision 0.2 - 4/11/88 - added units
* V200:   1/3/89 - mr --> cx

* calling argument declarations:
	INTEGER	cx, tlen

* internal variable declarations:
	CHARACTER*200	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND
	CHARACTER	VAR_UNITS*32
	INTEGER		TM_LENSTR1, maxlen, slen

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* initialize
	maxlen	= LEN( FULL_VAR_TITLE )

* title
	FULL_VAR_TITLE = VAR_TITLE( cx )
	tlen	       = TM_LENSTR1( FULL_VAR_TITLE )

* units
	FULL_VAR_TITLE = FULL_VAR_TITLE(:tlen) // ' ' //
     .		PAREN_SURROUND( VAR_UNITS(cx), slen )
	tlen = MIN( maxlen, tlen+1+slen)

* mods to title
	FULL_VAR_TITLE = FULL_VAR_TITLE(:tlen) //
     .		PAREN_SURROUND( VAR_TITLE_MOD(cx), slen )

* if it wont fit indicate that with a "*" at the end
	tlen = MIN( maxlen, tlen+slen )
	IF ( tlen .EQ. maxlen ) FULL_VAR_TITLE(maxlen:maxlen) = '*'

	RETURN
	END
