      SUBROUTINE ZOOM(X1,Y1,X2,Y2)
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
C**    @(#)zoom.f	1.1    3/10/88
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
C     CALL ZOOM(X1,Y1,X2,Y2)
C
C     THIS SUBROUTINE SELECTS THE BOUNDARIES OF THE AREA TO BE
C     DISPLAYED ON THE SCREEN OF THE TEKTERMINAL OR THE HP PLOT-
C     TER.  THIS AREA MUST BE SMALLER THAN, AND COMPLETELY CON-
C     TAINED WITHIN, THE OVERALL PLOTTING AREA DEFINED BY SUB-
C     ROUTINE SIZE.  SINCE THE AREA SELECTED FOR DISPLAY IS
C     SCALED TO FILL THE ENTIRE SCREEN, GREATER DETAIL IS VIS-
C     IBLE THAN WHEN THE ENTIRE PLOTTING AREA IS DISPLAYED.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X1     - X-COORDINATE OF THE FIRST CORNER OF THE SCREEN
C                 IN USER'S UNITS.
C        Y1     - Y-COORDINATE OF THE FIRST CORNER OF THE SCREEN
C                 IN USER'S UNITS.
C        X2     - X-COORDINATE OF THE OPPOSITE CORNER OF THE
C                 SCREEN IN USER'S UNITS.
C        Y2     - Y-COORDINATE OF THE OPPOSITE CORNER OF THE
C                 SCREEN IN USER'S UNITS.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     SCALE AND TRANSFORM THE COORDINATES OF THE FIRST CORNER OF
C     THE SCREEN.
C
      CALL TFORM(X1 * XSCALE,Y1 * YSCALE,ZTEMP)
      AMN = ANEW
      BMN = BNEW
C
C     SCALE AND TRANSFORM THE COORDINATES OF THE OPPOSITE CORNER
C     OF THE SCREEN.
C
      CALL TFORM(X2 * XSCALE,Y2 * YSCALE,ZTEMP)
      AMX = ANEW
      BMX = BNEW
C
C     FIND THE MINIMUM AND MAXIMUM VALUE OF EACH TRANSFORMED
C     COORDINATE.
C
      IF(AMN .GT. AMX)THEN
        TEMP = AMN
        AMN = AMX
        AMX = TEMP
      ENDIF
      IF(BMN .GT. BMX)THEN
        TEMP = BMN
        BMN = BMX
        BMX = TEMP
      ENDIF
C
C     RESTORE THE DEFAULT MINIMUM AND MAXIMUM ALLOWABLE VALUES
C     OF A AND B.
C
      AMIN = 0.0
      BMIN = 0.0
      AMAX = ASIZE
      BMAX = BSIZE
C
C     IF THE COORDINATES ARE IN THE PLOTTING AREA, SET THEM AS
C     THE MINIMUM AND MAXIMUM ALLOWABLE VALUES OF A AND B.
C
      IF(AMN .NE. AMX .AND. BMN .NE. BMX)THEN
        IF(AMN .GT. AMIN)AMIN = AMN
        IF(BMN .GT. BMIN)BMIN = BMN
        IF(AMX .LT. AMAX)AMAX = AMX
        IF(BMX .LT. BMAX)BMAX = BMX
      ENDIF
      ADIF = AMAX - AMIN
      BDIF = BMAX - BMIN
C
C     RECOMPUTE THE TEKTERMINAL CONVERSION FACTORS.
C
      IF(ADIF / TWIDTH .GT. BDIF / THIGHT)THEN
        PLTOTA = TSCLA / ADIF
        PLTOTB = TSCLB / ADIF * TWIDTH / THIGHT
      ELSE
        PLTOTA = TSCLA / BDIF * THIGHT / TWIDTH
        PLTOTB = TSCLB / BDIF
      ENDIF
C
C     RECOMPUTE THE HP PLOTTER CONVERSION FACTORS.
C
      PLTOHA = HPSCLA / ADIF
      PLTOHB = HPSCLB / BDIF
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
