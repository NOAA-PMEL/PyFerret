	INTEGER FUNCTION TM_LENSTR (line)

* determine the length of a character string as the position of the last 
* non-blank character (return 0 if all blanks)
* note: remember to declare function name INTEGER in calling program
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/21/85 - identical to LENSTR rev 0.00 apart from name
*
* Argument definition
	CHARACTER*(*)	line
	INTEGER		mright,i

* initialize: find highest possible right hand limit of string
	mright = LEN(line)

	DO 100 i = mright,1,-1
	IF (line(i:i) .NE. ' ') GOTO 200
 100	CONTINUE

* for all blanks
	TM_LENSTR = 0
	RETURN

* found non-blank
 200	TM_LENSTR = i
	RETURN
	END
