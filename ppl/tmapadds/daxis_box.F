      SUBROUTINE DAXIS_BOX

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
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOT]DAXIS.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection from interrupts in lower routines
*	  	  no explicit declarations - use FORT/WARN=NODECLARATIONS
* *jd*  - 8.18.95 Modification of DAXIS to draw only the plot border -- no
*		  tic marks or labels, for FILL & SHADE
* *jd*  -12.18.96 Add external statement for subroutine SCALE.  Is intrinsic
*                 function in f90 on linux.

#ifdef unix
	include 'tmap_pplv11inc/parampl5_dat.decl'
	include 'pplv11inc/PARAMPL5.DAT'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'
	include 'tmap_pplv11inc/labels_inc.decl'
	include 'pplv11inc/LABELS.INC'
	include 'tmap_pplv11inc/labcom_inc.decl'
	include 'pplv11inc/LABCOM.INC'
	include 'tmap_pplv11inc/hab_inc.decl'
	include 'pplv11inc/HAB.INC'
	include 'tmap_pplv11inc/lines_inc.decl'
	include 'pplv11inc/LINES.INC'
	include 'tmap_pplv11inc/pen_inc.decl'
	include 'pplv11inc/PEN.INC'
	include 'tmap_pplv11inc/pltl_inc.decl'
	include 'pplv11inc/PLTL.INC'
	include 'tmap_pplv11inc/mplot_inc.decl'
	include 'pplv11inc/MPLOT.INC'
	include 'tmap_pplv11inc/taxis_inc.decl'
	include 'pplv11inc/TAXIS.INC'
	include 'tmap_pplv11inc/tics_inc.decl'
	include 'pplv11inc/TICS.INC'
	include 'tmap_pplv11inc/tics2_inc.decl'
	include 'pplv11inc/TICS2.INC'
	include 'tmap_pplv11inc/axisl_inc.decl'
	include 'pplv11inc/AXISL.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:LABELS_INC.DECL'
	INCLUDE 'PPLV11INC:LABELS.INC'
	INCLUDE 'TMAP_PPLV11INC:LABCOM_INC.DECL'
	INCLUDE 'PPLV11INC:LABCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:HAB_INC.DECL'
	INCLUDE 'PPLV11INC:HAB.INC'
	INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
	INCLUDE 'PPLV11INC:LINES.INC'
	INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
	INCLUDE 'PPLV11INC:PLTL.INC'
	INCLUDE 'TMAP_PPLV11INC:MPLOT_INC.DECL'
	INCLUDE 'PPLV11INC:MPLOT.INC'
	INCLUDE 'TMAP_PPLV11INC:TAXIS_INC.DECL'
	INCLUDE 'PPLV11INC:TAXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS_INC.DECL'
	INCLUDE 'PPLV11INC:TICS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS2_INC.DECL'
	INCLUDE 'PPLV11INC:TICS2.INC'
	INCLUDE 'TMAP_PPLV11INC:AXISL_INC.DECL'
	INCLUDE 'PPLV11INC:AXISL.INC'
#endif
	CHARACTER IFX*20,IFY*20

* TMAP interrupt flag
#ifdef unix
	include 'tmap_pplv11inc/xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif

CCC     For linux install *jd* 12.96
        external scale

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
	
C
C     DRAW AXIS, LEFT, RIGHT Y AXIS THEN TOP, BOTTOM XAXIS
C

	IF(IWIND.GT.0)CALL WINDOW(0.,0.,0.,0.)

*	scale axis
      	call scale(1.,1.,0.,0.,0.,0.)

      IF(MYLEF.EQ.0.AND.IAXON(3).EQ.1)THEN
	call plot (xorg,yorg,0,0)
	call plot (xorg,yorg+ylen,1,0)
      END IF

	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(4).EQ.1)THEN
	call plot (xorg+xlen,yorg,0,0)
	call plot (xorg+xlen,yorg+ylen,1,0)
      ENDIF

	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(1).EQ.1)THEN
	call plot (xorg,yorg+ylen,0,0)
	call plot (xorg+xlen,yorg+ylen,1,0)
      END IF

	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(MXBOT.EQ.0.AND.IAXON(2).EQ.1)THEN
	call plot (xorg,yorg,0,0)
	call plot (xorg+xlen,yorg,1,0)
      END IF

* TMAP interrupt re-entry
 3333	CONTINUE

	RETURN
	END
