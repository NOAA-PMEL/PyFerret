	INTEGER FUNCTION ATOM_POS ( string, atom )

* find the position of the atom within the string
* if the atom is the last character in the string this represents an error

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/23/86
* revision 0.1 - 3/23/88  - bypass embedded quotated strings during search

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	CHARACTER*(*)	string
	CHARACTER*1	atom

* internal variable declarations:
	INTEGER		pos, slen

* initialize
	slen = LEN( string )
	pos  = 1

* loop top
 100	CONTINUE

* quotated string ?
	IF ( string(pos:pos) .EQ. '"' ) THEN
	   DO 110 pos = pos+1, slen
	      IF ( string(pos:pos) .EQ. '"' ) GOTO 200
 110	   CONTINUE
* ... unclosed quotation mark - flag it as a syntax error
	   ATOM_POS = atom_illegal_pos
	   RETURN

* found it ?
	ELSEIF ( string(pos:pos) .EQ. atom ) THEN
	   IF ( pos .EQ. slen ) THEN
	      ATOM_POS = atom_illegal_pos
	   ELSE
	      ATOM_POS = pos
	   ENDIF
	   RETURN
	ENDIF

* keep on looking ?
 200	IF ( pos .LT. slen ) THEN
	   pos = pos + 1
	   GOTO 100
	ENDIF

* no joy
	ATOM_POS = atom_not_found
	RETURN
	END
