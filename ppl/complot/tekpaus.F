      SUBROUTINE TEKPAS
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
C**    @(#)tekpaus.f	1.1    3/10/88
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
C     CALL TEKPAS
C
C     THIS SUBROUTINE SETS ALPHAS MODE AND SUSPENDS EXECUTION
C     TO ALLOW THE USER TO INSPECT THE PLOTTED OUTPUT.  EXECU-
C     TION RESUMES WHEN A CHARACTER IS ENTERED FROM THE KEY-
C     BOARD.  THIS CHARACTER IS DISCARDED.  ON THE HP PLOTTER,
C     IN ADDITION TO SETTING ALPHAS MODE, TEKPAS TERMINATES
C     THE CURRENT PLOT AND SENDS FURTHER OUTPUT TO THE TERMINAL.
C     TEKPAS HAS NO EFFECT ON THE ZETA AND GERBER PLOTTERS.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK THE PLOT TYPE.
C
      IF(PTYPE .EQ. 1 .OR. IABS(PTYPE) .GE. 2)THEN
C
C     PAUSE UNTIL THE USER RESPONDS.
C
        CALL PAGE
        CALL CHOUT(BEL,1)
        CALL CHOUT(BEL,1)
	CALL CHDMP
        CALL CHIN(IN,1,0)
      ENDIF
C
C     SET ALPHAS MODE.
C
      CALL ALPHAS
      RETURN
C
      END
