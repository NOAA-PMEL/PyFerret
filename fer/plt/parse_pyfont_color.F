	SUBROUTINE PARSE_PYFONT_COLOR (buff, red, grn, blu, status)


	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
        include 'pyfonts.cmn'

* Argument declarations 
	CHARACTER*(*) buff
	REAL*4	red, grn, blu
	INTEGER status

* Internal declarations 

	LOGICAL TM_DIGIT, MATCH4
	INTEGER TM_LENSTR1, TM_LENSTR, slen, s1, s2, icolor, nparm, 
     .		num_it, it_start(3), it_end(3), ncolor_string, 
     .		len_mchars, llen
	REAL	v1, v2, v3
	CHARACTER*20 errstrng

	PARAMETER (ncolor_string = 7)
	CHARACTER*12    colornames(ncolor_string)

* internal data declarations
	DATA colornames/'BLACK','RED','GREEN',
     .		    'BLUE','LIGHTBLUE','PURPLE','WHITE'/

	status = ferr_ok
	v1 = 0.
	v2 = 0.
	v3 = 0.

* Check for optional parentheses. 

	slen = TM_LENSTR1(buff)
	s1 = 1
	s2 = slen
	IF (buff(1:1) .EQ. '(' )THEN
           s1 = 2
           s2 = slen-1
           IF ( buff(slen:slen) .NE. ')' ) GOTO 5420
	ENDIF

	icolor = -1
	IF ( TM_DIGIT(buff(s1:s1) ) ) THEN 

* Read % values, convert to pen number or fraction.
	   nparm = 3
	   CALL parse_comma_list ( buff, s1, s2, nparm, num_it,
     .                it_start, it_end, status)
	   IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN
              nparm = 3 ! do nothing
	   ELSE IF (status .NE. ferr_ok .OR.
     .                num_it .LT. nparm) THEN

* Single color number? convert to pen # (needs to go get this from pplus iline bundle)
              IF (num_it .EQ. 1) THEN    ! single color-number?
		 READ (buff(it_start(1):it_end(1)), *) v1
	         icolor = v1
              ELSE
                 GOTO 5420
              ENDIF
	   ELSE

* 3 values, % rgb
              READ (buff(it_start(1):it_end(1)), *) v1
              IF (v1 .LT. 0. .OR. v1 .GT. 100) GOTO 5430
              red = v1/100.
              READ (buff(it_start(2):it_end(2)), *) v2
              IF (v2 .LT. 0. .OR. v2 .GT. 100) GOTO 5430
              grn = v2/100.
              READ (buff(it_start(3):it_end(3)), *) v3
              IF (v3 .LT. 0. .OR. v3 .GT. 100) GOTO 5430
              blu = v3/100.
	   ENDIF

	ELSE  ! not a digit. Is it a color name?

* name, compare with set list of colors
           llen = TM_LENSTR(buff(s1:s2))
           DO 1810 icolor = 1, ncolor_string
              len_mchars = TM_LENSTR(colornames(icolor)(:4))
              IF ( MATCH4(buff(s1:s2), llen,
     .                colornames(icolor),len_mchars ) ) GOTO 1820
 1810   CONTINUE
           GOTO 5420   ! unknown color
 1820	   CONTINUE
	ENDIF

* A color number. Get the current definition of color number icolor
	IF (icolor .GT. 0) THEN
           CALL GET_COLOR_RGB (icolor, v1,v2,v3, status)
           IF (status .NE. ferr_ok) GOTO 5420
           red = v1/100.
           grn = v2/100.
           blu = v3/100.
	ENDIF	   

 5000	RETURN

5420	errstrng = 'SET TEXT'
	slen = TM_LENSTR1(errstrng)
	llen = TM_LENSTR1(buff)
	CALL ERRMSG( ferr_syntax, status, 
     .			buff(:llen)//pCR//
     .			errstrng(:slen)//'/COLOR=(R,G,B) takes 3 arguments, '//
     .			'or valid color name or number', *5000 )
     
5430	errstrng = 'SET TEXT'
	slen = TM_LENSTR1(errstrng)
	llen = TM_LENSTR1(buff)
	CALL ERRMSG( ferr_out_of_range, status, 
     .			buff(:llen)//pCR//
     .			errstrng(:slen)//'/COLOR=(R,G,B) arguments must be between 0 and 100', 
     .			*5000 )
	END
