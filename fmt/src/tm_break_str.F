	SUBROUTINE TM_BREAK_STR ( instring, outstrings, numwords )
*
* This subroutine will "break up" the variable instring into 'numwords'
* strings and place them in the character array outstrings. The breaks
* occur at spaces and tabs.
*
* revision 0.00 - 05/15/87
*
* Arguments
	INTEGER		numwords
	CHARACTER*(*)	instring, outstrings(numwords)
*
	INTEGER		inlen, outlen, wcnt, nblnk, i, nchar

	inlen  = LEN(instring)
	outlen = LEN(outstrings(1))
	wcnt = 0
	nblnk = 1

* Locate next non-blank character (start of a word)
100	DO 200 i=nblnk,inlen
	  IF (instring(i:i) .NE. ' ') THEN
	    nchar = i
	    GOTO 300
	  ENDIF
200	CONTINUE
* no non-blank characters left
	GOTO 9900

* locate the next blank character (end of a word)
300	DO 400 i = nchar,inlen
	  IF (instring(i:i) .EQ. ' ') THEN
* end of word found
	    nblnk = i
	    GOTO 500
	  ENDIF
400	CONTINUE
* non blank characters left
	nblnk = inlen + 1

500	wcnt = wcnt + 1
	outstrings(wcnt) = instring(nchar:nblnk-1)
	IF (wcnt .LT. numwords) GOTO 100
* if at end of string loop at 100 will catch it

9900	RETURN
	END
