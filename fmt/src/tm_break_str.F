	SUBROUTINE TM_BREAK_STR ( instring, outstrings, numwords )
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
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
