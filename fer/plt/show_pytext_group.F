	SUBROUTINE SHOW_PYTEXT_GROUP (igrp)
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
        include 'xprog_state.cmn'
	include 'xrisc.cmn'
        include 'pyfonts.cmn'

	INTEGER	igrp

	INTEGER TM_LENSTR1, len, len0
	REAL	v1, v2, v3
	CHARACTER name*12

* List the PyFerret text settings for a text group: TICS, TITLE, LOGO, HLAB, VLAB, CONTOUR, CBAR

	IF (igrp.LT.itics .OR. igrp.GT.icbar) RETURN
	len = TM_LENSTR1(pyf_group_names(igrp))
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     '//pyf_group_names(igrp)(:len), 0)

c* ... Font
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'        Font:           '//pyf_fontname(igrp), 0)
* ... Color
	
* Convert rgb back to % values
	v1 = pyf_red(igrp)*100.
	v2 = pyf_grn(igrp)*100.
	v3 = pyf_blu(igrp)*100.
	WRITE (risc_buff, 2201) v1, v2, v3
 2201	FORMAT ('        Color (%RGB)    ', 2(f5.1,','), f5.1 )

	len = TM_LENSTR1(risc_buff)
	CALL GET_COLOR_NAME (v1, v2, v3, name, len0)
	IF (len0 .GT. 0) risc_buff = risc_buff(:len)//' ('//name(:len0)//')'
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			risc_buff, 0)

* ... Italic, Bold
	IF (pyf_italic(igrp) .EQ. 1) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'        Italic setting   ON', 0)

	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'        Italic setting   OFF', 0)

	ENDIF

	IF (pyf_bold(igrp) .EQ. 1) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'        Bold setting     ON', 0)

	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'        Bold setting     OFF', 0)

        ENDIF

	RETURN
	END
