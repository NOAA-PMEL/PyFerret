	SUBROUTINE TM_BREAK_LINES (instring,linestart,lineend,numlines)
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
*
* This subroutine will "break up" the variable instring into individual
* strings and place them in the character array outstrings. The breaks
* occur at the marker <NL>.  Return numlines, the number of lines found.
*
* based on tm_break_str.F
*
* Arguments
	INTEGER		numlines, linestart(*), lineend(*)
	CHARACTER*(*)	instring
       
*
        CHARACTER*4     newline
	INTEGER		STR_UPCASE, TM_LENSTR1,
     .                  icode, inlen, lincnt, nlin, i, nchar

* Initialize
	inlen  = TM_LENSTR1(instring)
	lincnt = 0
	nlin = 1
        numlines = 1
	linestart(numlines) = 1
	lineend(numlines) = inlen

* If the label ends in <NL> then ignore that.

        IF (inlen .GE. 4) THEN
           ICODE = str_upcase (newline, instring(inlen-3:inlen))  
           IF (newline .EQ. '<NL>') inlen = inlen-3
        ENDIF

* Locate next character

100	CONTINUE
        DO 200 i = nlin, inlen
	   nchar = i
	   GOTO 300
200	CONTINUE

* no characters left

	GOTO 9900

* locate the next <NL> (end of a line)

300	DO 400 i = nchar,inlen
	  IF (instring(i:i) .EQ. '<') THEN

* New line found?

           ICODE = str_upcase (newline, instring(i:i+3))  
           IF (newline .EQ. '<NL>') THEN
	      nlin = i + 4
              GOTO 500
            ENDIF
	  ENDIF
400	CONTINUE

* more characters left

	nlin = inlen + 1

500	lincnt = lincnt + 1
	linestart(lincnt) = nchar
	lineend(lincnt) = max(nchar,nlin-5)
	IF (nlin .LE. inlen) GOTO 100

* if at end of string loop at 100 will catch it

        numlines = lincnt
        lineend(lincnt) = inlen 

9900	RETURN
	END
