      SUBROUTINE SIZE(WIDTH,HEIGHT)
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
C**    @(#)size.F	1.1    3/10/88
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
C     PLOTTER DEPENDENT ROUTINE
C     CONTAINS PHYSICAL CHARACTERISTICS OF ALL LEGAL DEVICES.
C
C
C     CALL SIZE(WIDTH,HEIGHT)
C
C     THIS SUBROUTINE DEFINES THE DIMENSIONS OF THE TOTAL PLOT-
C     TING AREA.  SUBSEQUENT TO A CALL TO SIZE, PLOTTING IS AL-
C     LOWED ONLY WITHIN THESE LIMITS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        WIDTH  - TOTAL WIDTH OF THE PLOTTING AREA IN INCHES (OR
C                 MM).  WIDTH IS RESTRICTED TO A MAXIMUM OF 200
C                 INCHES ON THE ZETA PLOTTER AND 200 INCHES ON THE
C                 BENSON PLOTTER.
C        HEIGHT - TOTAL HEIGHT OF THE PLOTTING AREA IN INCHES
C                 (OR MM).  HEIGHT IS RESTRICTED TO A MAXIMUM OF
C                 34 INCHES ON THE ZETA PLOTTER AND 38 INCHES ON
C                 THE BENSON PLOTTER.
C
C     SIZE MAKES THE TEKTERMINAL AND PRINTRONIX PLOTTER APPEAR AS IF
C     THEY WERE WIDTH INCHES (OR MM) WIDE AND HEIGHT INCHES (OR
C     MM) HIGH.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
	include 'pplinc/GKSCM1.INC'
#else
	include 'pplinc:PLTCOM.DAT'
	include 'pplinc:GKSCM1.INC'
#endif
C
      IF(WIDTH .EQ. 0.0 .OR. HEIGHT .EQ. 0.0)RETURN
C
C     INSURE THAT THE PARAMETERS ARE POSITIVE AND CONVERT THEM
C     TO INTERNAL INCREMENTS.
C
      ANEW = FACTOR * ABS(WIDTH)
      BNEW = FACTOR * ABS(HEIGHT)
	IF(ANEW.EQ.ASIZE.AND.BNEW.EQ.BSIZE)RETURN
	ASIZE=ANEW
	BSIZE=BNEW
C
C     IN ACCORDANCE WITH THE PLOT TYPE, INSURE THAT THE DIMEN-
C     SIONS ARE NOT TOO LARGE.
C
      IF(PTYPE .EQ. 0 .OR. IABS(PTYPE) .GE. 2)THEN
        IF(ASIZE .GT. 200000.0)ASIZE = 200000.0
        IF(BSIZE .GT. 34000.0)BSIZE = 34000.0
      ENDIF
C
C     COMPUTE THE TEKTERMINAL CONVERSION FACTORS.
C
	IF(TTYPE.GT.0)THEN
      IF(ASIZE / TWIDTH .GT. BSIZE / THIGHT)THEN
        PLTOTA = TSCLA / ASIZE
        PLTOTB = TSCLB / ASIZE * TWIDTH / THIGHT
      ELSE
        PLTOTA = TSCLA / BSIZE * THIGHT / TWIDTH
        PLTOTB = TSCLB / BSIZE
      ENDIF
	ELSE
	PLTOTA=TSCLA/(TWIDTH*FACTOR)
	PLTOTB=TSCLB/(THIGHT*FACTOR)
	ENDIF
C
C     SET THE MINIMUM AND MAXIMUM ALLOWABLE VALUES OF A AND B.
C
      AMIN = 0.0
      BMIN = 0.0
      AMAX = ASIZE
      BMAX = BSIZE
	IF(PTYPE.GT.2)THEN
#ifdef core
	    call crinit
#else
	    call gkinit
#endif
	ENDIF
C
C     CALCULATE THE OFF-SCREEN FLAG.
C
      LOSFLG = 0
      IF(AOLD - 0.001 .GT. AMAX)LOSFLG = 1
      IF(AOLD + 0.001 .LT. AMIN)LOSFLG = 2
      IF(BOLD - 0.001 .GT. BMAX)LOSFLG = LOSFLG + 4
      IF(BOLD + 0.001 .LT. BMIN)LOSFLG = LOSFLG + 8
      RETURN
C
      END
