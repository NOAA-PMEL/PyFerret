	INTEGER FUNCTION TM_LENSTR1 (line)

* determine the length of a character string as the position of the last 
* non-blank character
* unlike TM_LENSTR this routine will return a length of 1 (vs. 0) if the string
* is all blanks

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 05/05/87 - based on TM_LENSTR rev. 0.00
*
* calling argument declarations:
	CHARACTER*(*) line

* internal variable declarations:
	INTEGER	i, mright

* initialize: find highest possible right hand limit of string
	mright = LEN(line)

	DO 100 i = mright,1,-1
	IF (line(i:i) .NE. ' ') GOTO 200
 100	CONTINUE

* for all blanks
	TM_LENSTR1 = 1
	RETURN

* found non-blank
 200	TM_LENSTR1 = i
	RETURN
	END
