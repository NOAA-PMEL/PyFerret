      SUBROUTINE WINDOW(XALOW,YALOW,XAHIGH,YAHIGH)
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
C**    @(#)window.f	1.1    3/10/88
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
C     CALL WINDOW(XLOW,YLOW,XHIGH,YHIGH)
C
C     THIS SUBROUTINE DEFINES A PLOTTING AREA THAT HAS DIFFERENT
C     BOUNDARIES THAN THE COMPLETE DISPLAY DEFINED BY SIZE.  AF-
C     TER A CALL TO THIS SUBROUTINE, ALL SUBSEQUENT PLOTTING MUST
C     LIE WITHIN BOTH SETS OF BOUNDARIES IN ORDER TO BE VISIBLE.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XLOW   - MINIMUM X-VALUE TO BE PLOTTED IN USER'S UNITS.
C        YLOW   - MINIMUM Y-VALUE TO BE PLOTTED IN USER'S UNITS.
C        XHIGH  - MAXIMUM X-VALUE TO BE PLOTTED IN USER'S UNITS.
C        YHIGH  - MAXIMUM Y-VALUE TO BE PLOTTED IN USER'S UNITS.
C
C
	include 'PLTCOM.DAT'
C
C     TRANSFER THE PARAMETERS TO NEW VARIABLES.
C
      XLOW = XALOW
      XHIGH = XAHIGH
      YLOW = YALOW
      YHIGH = YAHIGH
      IF(XHIGH .NE. XLOW .AND. YHIGH .NE. YLOW)THEN
C
C     SET THE WINDOW FLAG.
C
        WINDOF = .TRUE.
C
C     SET THE MINIMUM AND MAXIMUM ALLOWABLE VALUES OF X AND Y.
C
        XMIN = XLOW * XSCALE
        YMIN = YLOW * YSCALE
        XMAX = XHIGH * XSCALE
        YMAX = YHIGH * YSCALE
C
C     CHECK THE PARAMETERS TO MAKE SURE THAT THEY ARE VALID.
C
        IF(XMIN .GT. XMAX)THEN
          TEMP = XMIN
          XMIN = XMAX
          XMAX = TEMP
        ENDIF
        IF(YMIN .GT. YMAX)THEN
          TEMP = YMIN
          YMIN = YMAX
          YMAX = TEMP
        ENDIF
C
C     CALCULATE THE OFF-WINDOW FLAG.
C
        LOWFLG = 0
        IF(XOLD .GT. XMAX)LOWFLG = 1
        IF(XOLD .LT. XMIN)LOWFLG = 2
        IF(YOLD .GT. YMAX)LOWFLG = LOWFLG + 4
        IF(YOLD .LT. YMIN)LOWFLG = LOWFLG + 8
      ELSE
C
C     RESET THE WINDOW FLAG.
C
        WINDOF = .FALSE.
        OWFLG = 0
      ENDIF
      RETURN
C
      END
