      SUBROUTINE PLTEND
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
C**    @(#)pltend.f	1.1    3/10/88
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
C     CALL PLOTEND
C
C     THIS SUBROUTINE IS USED TO INDICATE THAT A DISPLAY HAS
C     BEEN COMPLETED.  ON THE TEKTERMINAL, EXECUTION IS SUS-
C     PENDED UNTIL THE USER RESPONDS BY ENTERING A CHARACTER
C     FROM THE KEYBOARD.  THEN THE SCREEN IS ERASED AND THE
C     ALPHA CURSOR IS MOVED TO THE UPPER LEFT-HAND CORNER OF
C     THE SCREEN.  ON THE PRINTRONIX PRINTER PLOTTER, THE BUFFER
C     IS DUMPED AND CONTROL IS RETURNED TO THE CALLING PROGRAM.
C     ON THE ZETA AND BENSON PLOT-
C     TERS, IF PLOTTING HAS BEEN DONE SINCE THE LAST CALL TO
C     PLOTEND OR ERASE, THE PEN IS LIFTED AND MOVED TO THE
C     LEFT MARGIN THREE INCHES ABOVE THE TOP OF THE LAST PLOT.
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
	IF(PTYPE.GE.3)CALL PPLGFLUSH
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
C     TERMINATE THE PLOT.
C
      CALL ERASE
      RETURN
C
      END
