	CHARACTER*132 FUNCTION TM_CMPRSS( instring )
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
