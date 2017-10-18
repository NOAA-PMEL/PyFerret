 	SUBROUTINE SYMBEL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
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
* V6.95 *acm* 3/15 Changes for PyFerret fonts. 
*                  SYMBEL_HERSHEY does Hershey fonts.
* V7.22 *acm* 10/17 Fix ticket 2566: if zero or negative character size, exit.

C V7.22 *acm* 10/17 Fix ticket 2566: if zero or negative character size, exit.

C
C	CALL SYMBEL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
C
C	THIS SUBROUTINE DRAWS A STRING OF CHARACTERS.
C
C	THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C	BELOW:
C        X,Y    - COORDINATES OF THE LOWER LEFT-HAND CORNER OF
C                 THE FIRST CHARACTER IN USERS UNITS.
C        DEGRES - ANGLE AT WHICH THE CHARACTERS ARE TO BE DRAWN
C                 IN DEGREES RELATIVE TO THE X-AXIS.
C        HEIGHT - HEIGHT OF THE CHARACTERS IN INCHES (OR MM).
C        NCHAR  - NUMBER OF CHARACTERS TO BE DRAWN.
C        STRING - ARRAY CONTAINING THE DISPLAY CODES OF THE
C                 CHARACTERS TO BE DRAWN.
C

* for Cairo fonts. Other font arguments are stored in common pyfonts.cmn:

* Calling arguments
	CHARACTER	STRING*(*)
	REAL		X,Y,DEGRES,HEIGHT
	INTEGER		NCHAR

* Local declarations
	LOGICAL		TM_HAS_STRING
	INTEGER		STR_SAME, TM_LENSTR, STR_UPCASE
	REAL		fontsiz, rotate, stx, sty, swid, shgt, xu, yu
     .			stx_pos, sty_pos, symcos, symsin
	INTEGER		ia, slen, ic, sub, sup, seb, sep, ss, se, flen, glen
	CHARACTER	buff1*512, buff2*512, buffs*512, upbuff*512
	INTEGER		codepoint, utf8strlen
	CHARACTER	utf8str*4, nstring*8, fontname*64, codeword*10
	REAL*4		red, grn, blu, fac
	INTEGER		igr, bold, italic, isiz, inote

	CHARACTER*265 err_msg
	CHARACTER*512 buff
	INTEGER status

	INCLUDE 'PLTCOM.DAT'
	INCLUDE 'pyfonts.cmn'
        INCLUDE 'errmsg.parm'


	IF (HEIGHT .LE. 0.) RETURN

* If font is "Hershey", continue with old-school PPLUS Hershey fonts.

	If (.NOT.pyfont) THEN
	   CALL SYMBEL_HERSHEY(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
	   GOTO 3000
	ENDIF

	IF (NCHAR .EQ. 0) RETURN
	rotate = degres
	IF (degres .NE. 0) rotate = -1.* degres

	slen = TM_LENSTR(string)
	buff = string(:slen)

* Specific font settings for the text group (if we're doing the logo, tic-marks, etc.)

	igr = igroup
	CALL GET_PYFONT_SETTINGS(igr, fontname, red, grn, blu, 
     .			italic, bold, isiz)

C Set font size in from the height 
        fontsiz = height

	IF (isiz .NE. 0) THEN
	   fac = FLOAT(ABS(isiz))
	   fac = MIN(4., fac)
	   if (isiz .LT. 0) fontsiz = fontsiz* (1.-fac*0.1)
	   if (isiz .GT. 0) fontsiz = fontsiz* (1.+fac*0.1)
	   fac = (1.-fac*0.2)
	   fac = (1.+fac*0.2)
	ENDIF

C Ignore any codes for Ferret-Hershey style Pen and Font changes 

	   slen = TM_LENSTR(string)
	   buff = string(:ABS(slen))

	   ia = INDEX (buff,'@')
	   inote = 0

	   DO WHILE (ia .GT. 0)
	      IF (NCHAR.GT.0) inote = 1
	      IF (ia .EQ. 1) THEN
	         buff = buff(ia+3:slen)
	      ELSE
	         buff1 = buff(1:ia-1)
	         buff = buff1(1:ia-1)//buff(ia+3:slen)
	      ENDIF
	      
	      slen = TM_LENSTR(buff)
	      ia = INDEX (buff,'@')
	   ENDDO
	   IF (inote .GT. 0) CALL WARN 
     .		('Non-Hershey fonts: ignoring Hershey pen or font codes') 

C The HTML would be &#nnn;  or &word; Because semicolon is used in Ferret, 
C look for <#nnn>  an HTML decimal code. e.g. <#916> is uppercase Delta
C or <& to start a word,  e.g. <&deg> for the degree sign.

	ss = INDEX (buff,'<#') + INDEX (buff,'<&')
	DO WHILE  (ss .GT. 0)
	   ia = INDEX (buff,'<#')
	   ic = INDEX (buff,'>')
	
	   IF (ia.GT.0 .AND. ic.GT.ia+2) THEN
	      ic = ic - 1
	      nstring = buff(ia+2:ic)
	      READ (nstring, *) codepoint

	      CALL CREATE_UTF8_STR(codepoint, utf8str, utf8strlen)
	      ic = ic+2
	      buff1 = buff(:ia-1)//utf8str(1:utf8strlen)//buff(ic:slen)
	      buff = buff1
	      slen = TM_LENSTR(buff)
	   ENDIF

C Is it a word, such as <&deg> ? If so, look up its decimal code

	   ia = INDEX (buff,'<&')
	   ic = INDEX (buff(ia:),'>') + ia-1
	
	   IF (ia.GT.0 .AND. ic.GT.ia+2) THEN
	      ic = ic - 1
	      nstring = buff(ia+2:ic)
	      READ (nstring, *) codeword
	      CALL SPECIAL_SYM_CODE (codeword, codepoint)
	      IF (codepoint .GT. 0) THEN  ! -1 means not found. For now, just leave that bad spec in.
	         CALL CREATE_UTF8_STR(codepoint, utf8str, utf8strlen)
	         ic = ic+2
	         buff1 = buff(:ia-1)//utf8str(1:utf8strlen)//buff(ic:slen)
	         buff = buff1
	         slen = TM_LENSTR(buff)
	      ENDIF
	   ENDIF
	   ss = INDEX (buff,'<#') + INDEX (buff,'<&')

	ENDDO

C Note any codes for Ferret-Hershey style Pen and Font changes 

	ia = INDEX(buff,'@')
	IF (ia .GT. 0) CALL WARN ('Non-Hershey fonts: not interpreting @pen and @font codes')

C Look for subscripts and superscript tags. 

	ia =  STR_UPCASE (upbuff, buff)
	sub = INDEX(upbuff, '<SUB>')
	sup = INDEX(upbuff, '<SUP>')
	seb = INDEX(upbuff,'</SUB>')
	sep = INDEX(upbuff,'</SUP>')
	IF (sub.GT.1 .AND. seb.EQ.0) GOTO 3300
	IF (sub.GT.1 .AND. seb.LT.sub) GOTO 3300
	IF (sup.GT.1 .AND. sep.EQ.0) GOTO 3300
	IF (sup.GT.1 .AND. sep.LT.sup) GOTO 3300

	IF  (sub.GT.0 .AND. sup.GT.0) THEN
	   ss = MIN(sub,sup)
	   IF (sub.LT.sup .AND. seb.GT.sup) GOTO 3300
	   IF (sub.LT.sup .AND. seb.GT.sep) GOTO 3300
	   IF (sup.LT.sub .AND. sep.GT.sub) GOTO 3300
	   IF (sup.LT.sub .AND. sep.GT.seb) GOTO 3300
	ELSE
	   ss = MAX(sub,sup)
           ENDIF

	buff1 = ''  ! Full-size portion of string
	buffs = ''  ! smaller sub- or super-script portion of string
	flen = 0
	glen = 0
	buff2 = buff

C This section returns the string length from swid

	IF (NCHAR .LE. 0) THEN

	   DO WHILE (ss.GT.0)
	      buff1 = buff1(:flen)//buff2(1:ss-1)
		 flen = ss-1
		 buff2 = buff2(ss+5:)
		 ia =  STR_UPCASE (upbuff, buff2)
		 se = INDEX(upbuff, '</SU')
	      buffs = buffs(:glen)//buff2(:se-1)
	      glen = se-1
	      buff2 = buff2(se+6:)

	      ia =  STR_UPCASE (upbuff, buff2)		 
	      sub = INDEX(upbuff, '<SUB>')
	      sup = INDEX(upbuff, '<SUP>')
	      seb = INDEX(upbuff,'</SUB>')
	      sep = INDEX(upbuff,'</SUP>')
	      IF (sub.GT.1 .AND. seb.EQ.0) GOTO 3300
	      IF (sup.GT.1 .AND. sep.EQ.0) GOTO 3300

	      IF  (sub.GT.0 .AND. sup.GT.0) THEN
	         ss = MIN(sub,sup)
		 IF (sub.LT.sup .AND. seb.GT.sup) GOTO 3300
		 IF (sub.LT.sup .AND. seb.GT.sep) GOTO 3300
		 IF (sup.LT.sub .AND. sep.GT.sub) GOTO 3300
		 IF (sup.LT.sub .AND. sep.GT.seb) GOTO 3300
	      ELSE
		 ss = MAX(sub,sup)
	      ENDIF
	   ENDDO

* Find the length of the total non-subscripted or superscripted string
	   glen = TM_LENSTR(buffs)
	   IF (glen .GT. 0) buff = buff1(:TM_LENSTR(buff1))//buff2(:glen)

	   swid = 0.
	   shgt = 0.
	   IF (TM_LENSTR(buff) .GT. 0) THEN
	      CALL FERTEXTSIZE (buff, fontname, fontsiz, 
     .           italic, bold, pyfnt_under, swid, shgt, status)
	      IF (status .EQ. 0) GOTO 3100
	   ENDIF

	   swid = swid/1000
	   shgt = shgt/1000
	   
	   x = swid

*  If there were subscripts and superscripts, those parts of the string are drawn at .75 size
	   IF (glen .GT. 0) THEN
	      swid = 0.
	      shgt = 0.
	      IF (glen .GT. 0) THEN
	         CALL FERTEXTSIZE (buffs(:glen), fontname, fontsiz*0.75, 
     .              italic, bold, pyfnt_under, swid, shgt, status)
	         IF (status .EQ. 0) GOTO 3100
	      ENDIF

	         swid = swid/1000
	         x = x + swid

	   ENDIF

	ELSE  ! Done with computing string width

C Draw the text

C If subscripts and/or superscripts do each part as it comes


	   symcos = COS(degres*1.7453292E-2) 
	   symsin = SIN(degres*1.7453292E-2) 

	   CALL USER_TO_INCH (x, y, stx_pos, sty_pos)
	   glen = 0

	   DO WHILE (ss.GT.0)
	      buff1 = buff2(1:ss-1)
	      flen = ss-1
	      buff2 = buff2(ss+5:)
	      ia =  STR_UPCASE (upbuff, buff2)
	      se = INDEX(upbuff, '</SU')
	      buffs = buff2(:se-1)
	      glen = se-1
	      buff2 = buff2(se+6:)

* Draw next section of full-sized text
	      stx = 1000.* stx_pos
	      sty = 1000.* sty_pos
	      CALL FERDRAWTEXT(buff1, fontname, fontsiz, 
     .             red, grn, blu, pyfnt_opq, stx, sty, 
     .             rotate, italic, bold, pyfnt_under, status)
	      IF (status .EQ. 0) GOTO 3200

* and mark the start of the next text to follow
	      swid = 0.
	      shgt = 0.
	      CALL FERTEXTSIZE (buff1, fontname, fontsiz, 
     .           italic, bold, pyfnt_under, swid, shgt, status)

	      IF (rotate.EQ.0) THEN
	         stx_pos = stx_pos + swid/1000  ! what about rotations?
	      ELSE
	         stx_pos = stx_pos + symcos*swid/1000 
	         sty_pos = sty_pos + symsin*swid/1000 
	      ENDIF

* Draw next superscript or subscript

	      IF (rotate.EQ.0) THEN
	         stx = 1000.* stx_pos
	         IF (ss .EQ. sub) sty = 1000.* (sty_pos - 0.5*fontsiz)
	         IF (ss .EQ. sup) sty = 1000.* (sty_pos + 0.5*fontsiz)
	      ELSE
	         IF (ss .EQ. sub) THEN
		    stx = 1000.* (stx_pos + symsin*0.5*fontsiz)
		    sty = 1000.* (sty_pos - symcos*0.5*fontsiz)
	         ENDIF
	         IF (ss .EQ. sup) THEN
		    stx = 1000.* (stx_pos - symsin*0.5*fontsiz)
		    sty = 1000.* (sty_pos + symcos*0.5*fontsiz)
		 ENDIF
	      ENDIF

	      CALL FERDRAWTEXT(buffs, fontname, 0.75*fontsiz, 
     .             red, grn, blu, pyfnt_opq, stx, sty, 
     .             rotate, italic, bold, pyfnt_under, status)
	      IF (status .EQ. 0) GOTO 3200

* and mark the start of the next text to follow
	      swid = 0.
	      shgt = 0.
	      CALL FERTEXTSIZE (buffs, fontname, 0.75*fontsiz, 
     .           italic, bold, pyfnt_under, swid, shgt, status)

	      IF (rotate.EQ.0) THEN
	         stx_pos = stx_pos + swid/1000
	      ELSE
	         stx_pos = stx_pos + symcos*swid/1000 
	         sty_pos = sty_pos + symsin*swid/1000 
	      ENDIF

	      ia =  STR_UPCASE (upbuff, buff2)
	      sub = INDEX(upbuff, '<SUB>')
	      sup = INDEX(upbuff, '<SUP>')
	      seb = INDEX(upbuff,'</SUB>')
	      sep = INDEX(upbuff,'</SUP>')
	      IF (sub.GT.1 .AND. seb.EQ.0) GOTO 3300
	      IF (sup.GT.1 .AND. sep.EQ.0) GOTO 3300

	      IF  (sub.GT.0 .AND. sup.GT.0) THEN
	         ss = MIN(sub,sup)
		 IF (sub.LT.sup .AND. seb.GT.sup) GOTO 3300
		 IF (sub.LT.sup .AND. seb.GT.sep) GOTO 3300
		 IF (sup.LT.sub .AND. sep.GT.sub) GOTO 3300
		 IF (sup.LT.sub .AND. sep.GT.seb) GOTO 3300
	      ELSE
		 ss = MAX(sub,sup)
                 ENDIF
	   ENDDO

* If there were subscripts or supserscripts, draw the last text
* or just draw the label string.

	   IF (glen .GT. 0) buff = buff2(:TM_LENSTR(buff2))

              stx = 1000.* stx_pos
              sty = 1000.* sty_pos
              CALL FERDRAWTEXT(buff, fontname, fontsiz, 
     .          red, grn, blu, pyfnt_opq, stx, sty, 
     .          rotate, italic, bold, pyfnt_under, status)
	   IF (status .EQ. 0) GOTO 3200
	
	   ENDIF


 3000	CONTINUE

	RETURN

 3100	err_msg = 'Error getting text size from PyFerret font call '
     .   //buff(:tm_lenstr(buff))
	CALL errmsg (ferr_nomessge, status, err_msg, *5002)
 3200	err_msg = 'Error drawing text with PyFerret Font call '
     .   //buff(:tm_lenstr(buff))

 3300	err_msg = 'Error in string: mismatch of <sub> or <sup> tags '
     .   //buff(:tm_lenstr(buff))

	CALL errmsg (ferr_nomessge, status, err_msg, *3000)

 5002	STOP

	END
