      SUBROUTINE PAGE
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
C**    @(#)page.f	1.1    3/10/88
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
C     CALL PAGE
C
C     THIS SUBROUTINE SETS ALPHAS MODE AND MOVES THE ALPHA CUR-
C     SOR ON A TEKTERMINAL TO THE UPPER LEFT-HAND CORNER OF THE
C     SCREEN (HOME POSITION).
C     PAGE HAS NO EFFECT ON THE ZETA AND BENSON OR PRINTRONIX PLOTTERS.
C
C
#ifdef unix
	include 'PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     SET THE PLOTTING MODE.
C
      PMODE = ALPHA
C
C     IF THE LAST POINT WAS NOT PLOTTED BECAUSE THE PEN WAS UP,
C     PLOT THE POINT NOW.
C
      IF(SAVEF)CALL XYZPLT
C
C     CHECK THE PLOT TYPE.
C
      IF(PTYPE .EQ. 0 .OR. PTYPE .EQ. -1.OR.PTYPE.GT.2)RETURN
C
C     CHECK THE TEKTERMINAL TYPE.
C
      IF(TTYPE .EQ. -4662 .OR. TTYPE .EQ. -4663)GO TO 10
C
C     MOVE TO HOME POSITION ON OTHER TEKTRONIX GRAPHICS TERMINALS.
C
        CALL CHOUT(HOME,5)
C
C     SELECT ALPHANUMERIC MODE ON OTHER TEKTRONIX GRAPHICS TER-
C     MINALS.
C
        CALL CHOUT(US,1)
      GO TO 30
C
C     CHECK THE GRAPHICS FLAG.
C
   10 IF(.NOT.GRAPHF) THEN
C
C     TRANSMIT THE 'PLOTTER ON' COMMAND TO THE TEKTRONIX PLOTTER.
C
        CALL CHOUT(TEKON,3)
      ENDIF
C
C     MOVE TO HOME POSITION ON THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(HOME,5)
C
C     SELECT ALPHANUMERIC MODE ON THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(US,1)
C
C     TRANSMIT THE 'PLOTTER OFF' COMMAND TO THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(TEKOFF,5)
C
C     WRITE OUT THE CHARACTER BUFFER.
C
   30 CALL CHDMP
C
C     RESET THE GRAPHICS FLAG.
C
      GRAPHF = .FALSE.
      RETURN
C
      END
