	SUBROUTINE PPL_AXES_RESTORE

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
* restore the PLOT+ axes after they have been turned off by PPL_AXES_OFF

* V450:  
* V541: 3/02  *acm* restore of PPL AXLABP settings too
*       3/25  Put the common block in a separate include file.
* v552 *acm* 3/03 new param IAXSET to restore axes after PLOT/AXES/set plot.

	include 'plot_setup.parm'
	include 'axis_inc.decl'
	include 'AXIS.INC'		! variable iaxon
        include 'ppl_ax_save.cmn'

* calling argument declarations

* local variable declarations:
	INTEGER i
	CHARACTER buff*16

* convert 4 axis integers to character as arg to the AXSET command
	buff = ' '
	WRITE (buff, '(4I3)' ) (save_iaxon(i), i=1, 4)

* restore axis state
	CALL PPLCMD ( from, line, 0, 'AXSET '//buff, 1, 1 )

* restore axis labelling

        IF (change_axlab) THEN
           WRITE (buff,92) save_labx, save_laby
  92	   FORMAT ('AXLABP ', i2, ',', I2 )
	   CALL PPLCMD (from, line, 0, buff, 1, 1)
        ENDIF

	IF ( time_axis .AND. change_axlab) THEN
	   WRITE (buff,94) save_labx, save_laby
  94	   FORMAT ('TXLABP ', i2, ',', I2 )
	   CALL PPLCMD (from, line, 0, buff, 1, 1)
	ENDIF

        iaxset = 0

	RETURN
	END
