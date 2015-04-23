	SUBROUTINE SET_PYTEXT_GROUP (igrp, buff1, buff2, status)
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
        include 'pyfonts.cmn'

	INTEGER	igrp, status
	CHARACTER*(*) buff1, buff2 

	INTEGER TM_LENSTR1, STR_SAME, llen, slen, s1, s2, nparm, 
     .		num_it,it_start(8), it_end(8), iit, i1
	REAL	v1
	CHARACTER arg*24, errstrng*20


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

* Check for and remove optional parentheses. 

	slen = TM_LENSTR1(buff1)
	s1 = 1
	s2 = slen
	IF (buff1(1:1) .EQ. '(' )THEN
	   s1 = 2
	   s2 = slen-1
	   IF ( buff1(slen:slen) .NE. ')' ) GOTO 5430
	   buff2 = buff1(s1:s2)
	   buff1 = buff2
	ENDIF
	
* Read arguments for /COLOR=, /FONT=, and /ISIZE=

	slen = TM_LENSTR1(buff1)
	nparm = 8
	CALL parse_comma_list ( buff1, 1, slen, nparm, num_it,
     .	     it_start, it_end, status)

* Loop over arguments to SET TEXT/[group]
	DO iit = 1, num_it
	   arg = buff1(it_start(iit):it_end(iit))

	   IF (STR_SAME(arg(1:3), "BOL") .EQ. 0) pyf_bold(igrp) = 1

	   IF (STR_SAME(arg(1:3), "ITA") .EQ. 0) pyf_italic(igrp)  = 1

	   IF (STR_SAME(arg(1:3), "COL") .EQ. 0) THEN  
	      CALL EQUAL_STR_LC( arg, buff2, status )
	      CALL PARSE_PYFONT_COLOR (buff2, igrp, pyf_red(igrp), 
     .	        pyf_grn(igrp), pyf_blu(igrp), status)
	      IF (status .NE. ferr_ok) GOTO 5000 ! error already issued
	   ENDIF

	   IF (STR_SAME(arg(1:3), "FON") .EQ. 0) THEN
	      CALL EQUAL_STR_LC( arg, buff2, status )
	      IF ( status .NE. ferr_ok ) GOTO 5440
	      IF ( buff2 .EQ. ' ' ) GOTO 5440
	      pyf_fontname(igrp) = buff2
	   ENDIF
	      
	   IF (STR_SAME(arg(1:3), "ISI") .EQ. 0) THEN
	      CALL EQUAL_VAL( arg, v1, status )
	      IF ( status .NE. ferr_ok ) GOTO 5450
	      i1 = INT(v1)
	      IF (i1.LT.-4 .OR. i1.GT.4) GOTO 5460
	      pyf_isiz(igrp) = i1
	   ENDIF

	ENDDO
	
	pyf_set(igrp) = .TRUE.
 5000	CONTINUE
 
	RETURN

* If there were errors, cancel any settings made above
 5100	CALL CANCEL_PYTEXT_GROUP (igrp)
	pyf_set(igrp) = .FALSE.
	RETURN
	
 5430	errstrng = 'SET TEXT'//pyf_group_names(igrp)
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_syntax, status, 
     .			buff1(:llen)//pCR//errstrng//
     .			' parentheses mismatch', *5100 )
 5440   errstrng = 'SET TEXT'//pyf_group_names(igrp)
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_invalid_command, status,
     .			errstrng(:slen)//'=what? ', *5100 )
 5450   errstrng = 'SET TEXT'//pyf_group_names(igrp)
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_invalid_command, status,
     .			errstrng(:slen)//'=what? ', *5100 )
 5460   errstrng = 'SET TEXT'//pyf_group_names(igrp)
	slen = TM_LENSTR1(errstrng)
        llen = TM_LENSTR1(buff1)
	CALL ERRMSG( ferr_out_of_range, status, 
     .                  buff1(:llen)//pCR//errstrng//
     .			' argument must be between -4 and 4', 
     .			*5100 )
	END
