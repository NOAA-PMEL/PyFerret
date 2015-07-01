	SUBROUTINE PUT_ANNOTATION (user, nouser, norm, xloc, yloc,
     .          halign, angle, size, nlab, labtxt, status)
	
	
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
* Make calls to PPLUS to set up the size angle and label for the DEFINE ANNOTATION 
* command. If the user has not set a label number, increment the count that is in 
* place for other moveable labels added by Ferret.

	include 'tmap_dims.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'

	LOGICAL user, norm, nouser
	INTEGER	nlab, status
	REAL	xloc, yloc, halign, angle, size
	CHARACTER*(*) labtxt

* local variable declarations:
	INTEGER TM_LENSTR1, rlen, nchar
	CHARACTER TM_FMT*14, buff*20
	
* Count of (PLOT+ movable labels) or use the nlab given by the user
	IF (nlab .EQ. 0) THEN
 	   nlabs_on = nlabs_on + 1
	   nlab = nlabs_on
	ENDIF

* Send label size to pplus
	WRITE ( ppl_buff, 3000 ) nlab, size
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

* Send rotation angle to pplus
	IF (angle .NE. 0.) THEN
	   WRITE ( ppl_buff, 3100 ) nlab, angle
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ENDIF

* Send LABS command with qualifier, position, justify setting, and label

	WRITE ( ppl_buff, 3200 ) nlab
	IF (user) WRITE ( ppl_buff, 3300 ) nlab
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(xloc, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(yloc, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//','//TM_FMT(halign, 14, 16, nchar)
	rlen = TM_LENSTR1(ppl_buff)

	ppl_buff = ppl_buff(:rlen)//', '//labtxt
	rlen = TM_LENSTR1(ppl_buff)

	CALL PPLCMD ( from, line, 0, ppl_buff(:rlen), 1, 1 )

	RETURN
	
 3000	FORMAT('HLABS ',I3,1X,F6.3 )
 3100	FORMAT ('RLABS ',I4, ',', F8.3)
 3200	FORMAT('LABS/NOUSER ', I3)
 3300	FORMAT('LABS/USER ', I3)

	END