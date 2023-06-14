      SUBROUTINE COLOR(ICODE)
C**
C**    @(#)color.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     CALL COLOR(ICODE)
C
C     THIS SUBROUTINE SELECTS A PEN ON THE TEKTRONIX 4663 PLOTTER,
C     THE HP PLOTTER, AND THE ZETA PLOTTER AND CONTROLS THE SIX-
C     POSITION TURRET HEAD ON THE GERBER PLOTTER.  IT CAN BE USED
C     TO CHANGE THE COLOR OF THE INK USED FOR PLOTTING.
C
C
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
*	Modified J Davison JISAO/PMEL/TMAP 3.23.89
*	To accomodate GKS changes to bundled line attributes
 
*  8/2001  acm  for PLOT/SYMBOL=DOT
*  Call GSPMI for polymarker with call to GSPLI for polyline
*      *acm* 10/01  remove VMS includes
*  12/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals

	integer	icode

	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
        include 'makedot.cmn' ! for OLDCOLORID

C     CHECK THE PLOT TYPE.

	IF (PTYPE .LT. 3) THEN
	   IF(COLOUR.LT.0.OR.COLOUR.GT.15)COLOUR=15

	ELSE

C	GKS COLOR AND SETUP

	   COLOUR=MAX(ICODE,1)

	   IF ( .NOT. GKSOPN )
     .      STOP 'COLOR called when GKSOPN is .FALSE.'

	   IF ( OLDCOLORID .EQ. COLOUR ) THEN
*           pen is not actually changing, so nothing to do
	      RETURN
	   ELSE
	      OLDCOLORID = COLOUR
	   ENDIF

*       Force the plotting of accumulated lines (first) 
*       and markers (second) using the old color
	   CALL PPLGFLUSH
	   CALL MAKEDOTFLUSH


*       Set the new color for lines and markers
	   CALL FGD_GSPLI(COLOUR)

	ENDIF

	RETURN
	END
