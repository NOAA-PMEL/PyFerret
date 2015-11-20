	SUBROUTINE SET_PYTEXT_GROUP (igrp, buff1, status)
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
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'errmsg.parm'
	include	'slash.parm'
	include 'xprog_state.cmn'
        include 'pyfonts.cmn'

	INTEGER	igrp, status
	CHARACTER*(*) buff1 

	INTEGER TM_LENSTR1, STR_SAME, llen, slen, ii, loc
	REAL	val_buf
	CHARACTER arg*24, errstrng*20

* Settings for all text

      IF (igrp .EQ. fnt_all) THEN

c* ... SET TEXT/FONT
	IF ( qual_given( slash_set_text_font ) .GT. 0 ) THEN
           loc = qual_given( slash_set_text_font )
	   IF ( loc.GT. 0 ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff1, status )
              IF ( status .NE. ferr_ok ) RETURN
	      IF ( buff1 .EQ. ' ' ) GOTO 5440
	      pyfnt_fontname = buff1
	   ENDIF
	ENDIF

* ... SET TEXT/COLOR
*     % rgb or color names black,red,green,blue,lightblue,purple,white or color #

	IF ( qual_given( slash_set_text_color ) .GT. 0 ) THEN
	   loc= qual_given( slash_set_text_color )
	
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff1, status )
           IF ( status .NE. ferr_ok ) GOTO 5100
	   CALL PARSE_PYFONT_COLOR (buff1, pyfnt_red, 
     .         pyfnt_grn, pyfnt_blu, status)
           IF ( status .NE. ferr_ok ) GOTO 5100
	ENDIF 

* ... SET TEXT/ITALIC
	IF ( qual_given( slash_set_text_italic ) .GT. 0 ) THEN	
	   pyfnt_italic = 1
	ENDIF

* ... SET TEXT/BOLD
	IF ( qual_given( slash_set_text_bold ) .GT. 0 ) THEN	
	   pyfnt_bold = 1
	ENDIF

* ... SET TEXT/ISIZ
	IF ( qual_given( slash_set_text_isiz ) .GT. 0 ) THEN	
           loc = qual_given(slash_set_text_isiz)

*       Get the isize value
           CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     val_buf, status )
           IF ( status .NE. ferr_ok ) GOTO 5450
	   
           ii = INT(val_buf)
           IF (ii.LT.-4 .OR. ii.GT.4) GOTO 5460
           pyfnt_isiz = ii
	ENDIF

* Set the group settings to the default values and unset pyf_set
* Then the groups will follow the global setting.

	DO 1000 ii = 1, ngp
	   pyf_italic(ii) = 0
	   pyf_bold(ii) = 0
	   pyf_isiz(ii) = 0
	   pyf_red(ii) = 0
	   pyf_grn(ii) = 0
	   pyf_blu(ii) = 0
	   pyf_fontname(ii) = pyf_font_dflt(ii)
	   pyf_set(ii) = .FALSE.
 1000	CONTINUE

      ELSE  ! group settings

* If there were not previous settings on this group, first apply the
* current global settings to this group

	IF (.NOT. pyf_set(igrp)) THEN
	   pyf_italic(igrp) = pyfnt_italic
	   pyf_bold(igrp) = pyfnt_bold
	   pyf_isiz(igrp) = pyfnt_isiz
	   pyf_red(igrp) = pyfnt_red
	   pyf_grn(igrp) = pyfnt_grn
	   pyf_blu(igrp) = pyfnt_blu
	   pyf_fontname(igrp) = pyfnt_fontname
	ENDIF
	
* Read arguments for /COLOR=, /FONT=, and /ISIZE=

c* ... SET TEXT/FONT
	IF ( qual_given( slash_set_text_font ) .GT. 0 ) THEN
           loc= qual_given( slash_set_text_font )
	   IF ( loc.GT. 0 ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff1, status )
              IF ( status .NE. ferr_ok ) RETURN
	      IF ( buff1 .EQ. ' ' ) GOTO 5440
	      pyf_fontname(igrp) = buff1
	   ENDIF
	ENDIF

* ... SET TEXT/COLOR
*     % rgb or color names black,red,green,blue,lightblue,purple,white or color #

	IF ( qual_given( slash_set_text_color ) .GT. 0 ) THEN
	   loc= qual_given( slash_set_text_color )
	
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff1, status )
           IF ( status .NE. ferr_ok ) GOTO 5100
	   CALL PARSE_PYFONT_COLOR (buff1, pyf_red(igrp), 
     .         pyf_grn(igrp), pyf_blu(igrp), status)
           IF ( status .NE. ferr_ok ) GOTO 5100
	ENDIF 


* ... SET TEXT/ITALIC
	IF ( qual_given( slash_set_text_italic ) .GT. 0 ) THEN	
	   pyf_italic(igrp) = 1
	ENDIF

* ... SET TEXT/BOLD
	IF ( qual_given( slash_set_text_bold ) .GT. 0 ) THEN	
	   pyf_bold(igrp) = 1
	ENDIF

* ... SET TEXT/ISIZ
	IF ( qual_given( slash_set_text_isiz ) .GT. 0 ) THEN	
           loc = qual_given(slash_set_text_isiz)

*       Get the isize value
           CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     val_buf, status )
           IF ( status .NE. ferr_ok ) GOTO 5450
	   
           ii = INT(val_buf)
           IF (ii.LT.-4 .OR. ii.GT.4) GOTO 5460
           pyf_isiz(igrp) = ii
	ENDIF

         pyf_set(igrp) = .TRUE.

      ENDIF  ! group vs global settings
	
 5000 CONTINUE
 
      RETURN

* If there were errors, cancel any settings made above
 5100	CALL CANCEL_PYTEXT_GROUP (igrp)
	pyf_set(igrp) = .FALSE.
	RETURN

 5440   CALL ERRMSG( ferr_invalid_command, status,
     .			'SET TEXT/FONT=what? See Unix system fonts ', *5000 )
 5450   errstrng = 'SET TEXT'
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_invalid_command, status,
     .			errstrng(:slen)//'=what? ', *5100 )
 5460   errstrng = 'SET TEXT'
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_out_of_range, status, 
     .                  buff1(:llen)//pCR//errstrng//
     .			' argument must be between -4 and 4', 
     .			*5100 )
	END
