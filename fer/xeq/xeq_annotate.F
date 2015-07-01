	SUBROUTINE XEQ_ANNOTATE

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
* Put an annotation (moveable label) onto an existing plot

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP/SDIG 
*
* revision 0.0 - 4/6/2015

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	
* local variable declarations:
	LOGICAL user, norm, nouser
	INTEGER	TM_LENSTR1,
     .		rlen, item, nchar, status
	REAL	xloc, yloc, halign, angle, size

	CHARACTER TM_FMT*14, buff*20, labtxt*2048

* initialize 

	size = 0.2

* Get the text
	IF (num_items .NE. 1) GOTO 5500
	item = 1
	labtxt = cmnd_buff(item_start(item):item_end(item))

* Get settings

	CALL SET_ANNOTATION (user, nouser, norm, xloc, yloc, halign, 
     .         angle, size, status)
	IF (status .NE. ferr_ok) GOTO 5000

	IF (nouser .OR. norm) THEN
	   buff = 'NOUSER'
	ELSE
	   buff = 'USER'
	ENDIF

* Create the %LABEL command

	ppl_buff = '%LABEL/'//buff
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//' '//TM_FMT(xloc, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(yloc, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(halign, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(angle, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(size, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//', '//labtxt
	rlen = TM_LENSTR1(ppl_buff)

* ... pass a the command to PPLUS

c	print *, ppl_buff(:rlen)
	CALL PPLCMD (' ',' ',0,ppl_buff(:rlen),1,1)

 5000	RETURN
 5500	CALL ERRMSG( ferr_invalid_command, status,
     .		'ANNOTATE command missing text for the annotation', *5000 )

	END
