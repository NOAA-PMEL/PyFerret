      SUBROUTINE escape_for_xml(string, outstring, slen)
      
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
* When writing xml, the string inside the tags may not contain > < or &.
* Replace these characters in the input string with the appropriate 
* replacement string, as outlined here:
*
* http://www.w3.org/TR/REC-xml/
* The ampersand character (&) and the left angle bracket (<) MUST NOT 
* appear in their literal form, except when used as markup delimiters, 
* or within a comment, a processing instruction, or a CDATA section. If 
* they are needed elsewhere, they MUST be escaped using either numeric 
* character references or the strings " &amp; " and " &lt; " respectively. 
* The right angle bracket (>) may be represented using the string " &gt; ", 
* and MUST, for compatibility, be escaped using either " &gt; " or a character 
* reference when it appears in the string " ]]> " in content, when that string 
* is not marking the end of a CDATA section. 

* V631 11/09 *acm* 

* calling argument declaration
      CHARACTER*(*) string, outstring 
      INTEGER slen 

* Local declarations
      INTEGER TM_LENSTR1, i, nchar, lenrepl(3), blen
      CHARACTER*1 rchar(3)
      CHARACTER*5 repl(3)
      
      DATA rchar(1)  /'<'/
      DATA rchar(2)  /'>'/
      DATA rchar(3)  /'&'/
      DATA repl(1)  /'&lt;'/
      DATA repl(2)  /'&gt;'/
      DATA repl(3)  /'&amp;'/

      DATA lenrepl/4,4,5/

      nchar = 3
      slen = TM_LENSTR1(string)
      loc = 1

      outstring(1:1) = string(loc:loc)
      blen = 1

      DO 90 i = 1, nchar
         IF (string(loc:loc) .EQ. rchar(i)) THEN
            outstring = repl(i)
            blen = lenrepl(i)
         ENDIF
  90     CONTINUE

      loc = loc + 1

      DO WHILE (loc .LE. slen)
         DO 100 i = 1, nchar
            IF (string(loc:loc) .EQ. rchar(i)) THEN
               outstring = outstring(1:blen)//repl(i)
               blen = blen + lenrepl(i)
               GOTO 200
            ENDIF
  100    CONTINUE

*  Not replaced, just keep the orig. character

         outstring = outstring(1:blen)//string(loc:loc)
         blen = blen + 1

  200    CONTINUE

         loc = loc + 1

      ENDDO

      slen = blen
      
      RETURN
      END

