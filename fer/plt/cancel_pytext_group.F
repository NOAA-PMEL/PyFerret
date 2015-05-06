	SUBROUTINE CANCEL_PYTEXT_GROUP (igrp)
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

* V695  *acm*  5/15 on canceling group settings, restore to the current global setting

        include 'tmap_dims.parm'
	include	'slash.parm'
	include 'xprog_state.cmn'
        include 'pyfonts.cmn'

	INTEGER	igrp

	INTEGER i
	LOGICAL iall, is_set

* Restore default settings

* Should the logo pyf_set be set to .TRUE.? Should it be set to the overall text font, 
* or its own text font?

      IF (igrp .EQ. fnt_all) THEN
        IF ( qual_given( slash_canc_text_all ) .GT. 0) iall = .TRUE.

	IF (num_qualifiers .EQ. 0) iall = .TRUE.

c* ... CANCEL TEXT/FONT
	IF (iall .OR. qual_given( slash_canc_text_font ) .GT. 0) THEN
           pyfnt_fontname = pyfnt_font_dflt
	   
           DO 1010 i = 1, ngp
	      pyf_fontname(i) = pyf_font_dflt(i)
 1010      CONTINUE
	ENDIF

* ... CANCEL TEXT/COLOR
*     Restore to black

	IF (iall .OR. qual_given( slash_canc_text_color ) .GT. 0) THEN
           pyfnt_red = 0.
	   pyfnt_grn = 0.
	   pyfnt_blu = 0.
           DO 1020 i = 1, ngp
	      pyf_red(i) = pyfnt_red
	      pyf_grn(i) = pyfnt_grn
	      pyf_blu(i) = pyfnt_blu
 1020      CONTINUE
	ENDIF

* ... CANCEL TEXT/ITALIC

	IF (iall .OR. qual_given( slash_canc_text_italic ) .GT. 0) THEN
           pyfnt_italic = 0
           DO 1030 i = 1, ngp
	      pyf_italic(i) = pyfnt_italic
 1030      CONTINUE
	ENDIF

* ... CANCEL TEXT/BOLD

	IF (iall .OR. qual_given( slash_canc_text_bold) .GT. 0) THEN
           pyfnt_bold = 0
           DO 1040 i = 1, ngp
	      pyf_bold(i) = pyfnt_bold
 1040      CONTINUE
	ENDIF

* ... CANCEL TEXT/ISIZ

	IF (iall .OR. qual_given( slash_canc_text_isiz) .GT. 0) THEN
           pyfnt_isiz = 0
           DO 1050 i = 1, ngp
	      pyf_isiz(i) = pyfnt_isiz
 1050      CONTINUE
	ENDIF

* ...   Mark all groups as unset

	DO 1060 i = 1, ngp
           pyf_set(i) = .FALSE.
 1060	CONTINUE


* PyFont text groupings.  Restore group to the current global settings.
* Compare with global settings. If they match the global settings, 
* unset pyfer_set

      ELSE

	IF (num_qualifiers .EQ. 0) iall = .TRUE.
      
c* ... CANCEL TEXT/FONT
	IF (qual_given( slash_canc_text_font ).GT.0 .OR. iall) THEN
           pyf_fontname(igrp) = pyfnt_fontname
	ENDIF

* ... CANCEL TEXT/COLOR
*     Restore to black

	IF (qual_given( slash_canc_text_color ).GT.0 .OR. iall) THEN
           pyf_red(igrp) = pyfnt_red
           pyf_grn(igrp) = pyfnt_grn
           pyf_blu(igrp) = pyfnt_blu
	ENDIF

* ... CANCEL TEXT/ITALIC

	IF (qual_given( slash_canc_text_italic ).GT.0 .OR. iall) THEN
           pyf_italic(igrp) = pyfnt_italic
	ENDIF

* ... CANCEL TEXT/BOLD

	IF (qual_given( slash_canc_text_bold).GT.0 .OR. iall) THEN
           pyf_bold(igrp) = pyfnt_bold
	ENDIF

* ... CANCEL TEXT/ISIZ

	IF (qual_given( slash_canc_text_isiz).GT.0 .OR. iall) THEN
           pyf_isiz(igrp) = pyfnt_isiz
	ENDIF

* Do all the group settings match the current gloabl ones?  If so 
* set pyf_set accordingly

	IF (iall) THEN
	   pyf_set(igrp) = .FALSE.
	ELSE
	   is_set = .FALSE.
	   IF (pyf_fontname(igrp) .NE. pyfnt_fontname) is_set = .TRUE.
	   IF (pyf_red(igrp) .NE. pyfnt_red)  is_set = .TRUE.
	   IF (pyf_grn(igrp) .NE. pyfnt_grn)  is_set = .TRUE.
	   IF (pyf_blu(igrp) .NE. pyfnt_blu)  is_set = .TRUE.
	   IF (pyf_italic(igrp) .NE. pyfnt_italic)  is_set = .TRUE.
	   IF (pyf_bold(igrp) .NE. pyfnt_bold)  is_set = .TRUE.
	   IF (pyf_isiz(igrp) .NE. pyfnt_isiz)  is_set = .TRUE.
	   pyf_set(igrp) = is_set
	ENDIF

      ENDIF

      RETURN
      END
      