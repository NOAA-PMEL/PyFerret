	CHARACTER*132 FUNCTION TM_CMPRSS( instring )
*
* Takes multiple blanks and tabs in a string and replaces them with a
* single blank.
*
*	written by M. Verschell  04/03/87 for PMEL/TMAP
*
* revision 0.00 - 04/03/87
*
* argument
	CHARACTER*(*)	instring

* local definitions
	CHARACTER*1	tab
	INTEGER		inlen, i, pos, nblnk, blnk

	tab = CHAR(09)

	TM_CMPRSS =  ' '
	inlen = LEN(instring)
	pos = 0
	blnk = 1

* Find next non-blank or non-tab character
 100	DO i = blnk,inlen
	  IF (instring(i:i) .NE. ' ' .AND. instring(i:i) .NE. tab) THEN
* Store postion of non-blank character in input string
	    nblnk = i
* Pos is where that character goes in output string
	    pos = pos + 1
	    GOTO 200
	  ENDIF
	ENDDO
* If we get this far all that's left is trailing blanks (tabs)
	GOTO 9990

* transfer non-blank characters to output one-at-a-time until blank/tab
 200	DO i = nblnk,inlen
	  IF (instring(i:i) .NE. ' ' .AND. instring(i:i) .NE. tab) THEN
	    TM_CMPRSS(pos:pos) = instring(i:i)
	    pos = pos + 1
	  ELSE
* first blank character
	    TM_CMPRSS(pos:pos) = ' '
	    blnk = i
	    GOTO 100
	  ENDIF
	ENDDO
* done with string
9990	RETURN
	END	    
