	INTEGER FUNCTION ALG_CLASS( ctr )

* subroutine for lexical analyzer
* determine chatacter class of character
* note: the routine could be made simpler and faster by simply providing a
*	DATA statement associating each character number (ICHAR('C')) with a
*	class.   ... hurried development ...

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 11/24/89
* V495: 7/7/98 - add single quotes as another character class (11) (old 11->12)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declaration:
	CHARACTER*1	ctr

* internal parameter declarations:
	CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9) )

* classes of characters
*	C1 - + -
*	C2 - digits
*	C3 - letters _ _
*	C4 - [
*	C5 - ]
*	C6 - * / ^ , (
*	C7 - )
*	C8 - blank tab
*	C9 - E e
*	C10- .
*       C11- '
*	C12- all others

	IF     ( ctr.EQ.'+' .OR.  ctr.EQ.'-' ) THEN
	   ALG_CLASS = 1
	ELSEIF ( ctr.GE.'0' .AND. ctr.LE.'9' ) THEN
	   ALG_CLASS = 2
	ELSEIF ( ctr.EQ.'E' .OR.  ctr.EQ.'e' ) THEN
	   ALG_CLASS = 9
	ELSEIF ( ( ctr.GE.'A' .AND. ctr.LE.'Z' )
     .	    .OR. ( ctr.GE.'a' .AND. ctr.LE.'z' )
     .	    .OR. ( ctr.EQ.'_'  .OR. ctr.EQ.'_' )	) THEN
	   ALG_CLASS = 3
	ELSEIF ( ctr.EQ.'[' ) THEN
	   ALG_CLASS = 4
	ELSEIF ( ctr.EQ.']' ) THEN
	   ALG_CLASS = 5
	ELSEIF ( ctr.EQ.'*' .OR. ctr.EQ.'/' .OR. ctr.EQ.'^'
     .	    .OR. ctr.EQ.',' .OR. ctr.EQ.'('		) THEN
	   ALG_CLASS = 6
	ELSEIF ( ctr.EQ.')' ) THEN
	   ALG_CLASS = 7
	ELSEIF ( ctr.EQ.' ' .OR. ctr.EQ.tab ) THEN
	   ALG_CLASS = 8
	ELSEIF ( ctr.EQ.'.' ) THEN
	   ALG_CLASS = 10
	ELSEIF ( ctr.EQ.'\'' ) THEN
	   ALG_CLASS = 11
	ELSE
	   ALG_CLASS = 12
	ENDIF

	RETURN
	END
