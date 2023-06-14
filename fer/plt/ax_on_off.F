	SUBROUTINE ax_on_off (iax, overlay)
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
* August 2001
* call AXSET to turn axes off or on according to argument iax (0=off, 1=on):
*  iax(1) TOP	 
*  iax(2) BOTTOM
*  iax(3) LEFT
*  iax(4) RIGTH

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* v540 *acm*  2/02 save the state of the axes; so can restore PPL AXSEST settings
*             3/25  Put the common block in a separate include file.
* v552  *acm* 3/03 new param IAXSET to restore axes after PLOT/AXES/set plot.
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* v744 *acm* 11/18 Issue 1786: graticule lines & axes drawn if PLOT/OVER/AXES given
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'plot_setup.parm'
        include 'xplot_setup.cmn'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'ppl_ax_save.cmn'

	INTEGER		iax(*)
	LOGICAL		overlay

	INTEGER		i, indep_dim, dep_dim, labx, laby, sum
	CHARACTER*25	buff

* save previous axis state (top,bottom,left,right)

	DO i = 1, 4
	   save_iaxon(i) = iaxon(i)
 	ENDDO
        save_labx = labelx
        save_laby = labely

	WRITE (buff,90) (iax(i), i=1,4)
  90	FORMAT ('AXSET ', 3(i2,','), I2 )
	CALL PPLCMD (from, line, 0, buff, 1, 1 )


*  turn axis labels off or on

	IF (iax(1) .EQ. 0  .AND. iax(2) .EQ. 0) CALL PPLCMD 
     .         ( from, line, 0, 'XLAB', 1, 1 )

	IF (iax(3) .EQ. 0  .AND. iax(4) .EQ. 0) CALL PPLCMD 
     .         ( from, line, 0, 'YLAB', 1, 1 )

*  When only top or right axis plotted, labels wouldnt appear. 
*  Turn them on.

        labx = labelx
        laby = labely
        change_axlab = .FALSE.
        IF (iax(1) .EQ. 1  .AND. iax(2) .EQ. 0) THEN
           labx = 1
           change_axlab = .TRUE.
	ENDIF
        IF (iax(3) .EQ. 0  .AND. iax(4) .EQ. 1) THEN
           laby = 1
           change_axlab = .TRUE.
        ENDIF
        IF (change_axlab) THEN
           WRITE (buff,92) labx, laby
  92	   FORMAT ('AXLABP ', i2, ',', I2 )
	   CALL PPLCMD (from, line, 0, buff, 1, 1)
        ENDIF

*  What if one is a time axis?  

	indep_dim = plot_axis(1)
	dep_dim = plot_axis(2)
	time_axis = ( (indep_dim.EQ.t_dim .OR. indep_dim.EQ.f_dim)
     .    .OR. (dep_dim.EQ.t_dim .OR. dep_dim.EQ.t_dim) )

	IF ( time_axis .AND. change_axlab) THEN
	   WRITE (buff,94) labx, laby
  94	   FORMAT ('TXLABP ', i2, ',', I2 )
	   CALL PPLCMD (from, line, 0, buff, 1, 1)
	ENDIF

        iaxset = 1

* Is this an overlay plot where they requested axes be drawn?
* Store a flag (AXIS.INC).

	IF (overlay) THEN
	   axes_on_overlay = .FALSE.
	   sum = 0
	   DO i = 1, 4
	     sum = sum + iax(i)
	   ENDDO
	   IF (sum .GT. 0) axes_on_overlay = .TRUE.
	ENDIF
	
	RETURN
	END
