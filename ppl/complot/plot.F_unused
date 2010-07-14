      SUBROUTINE PLOT(X,Y,IPEN,IMARK)
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
C**    @(#)plot.f	1.1    3/10/88
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
C     CALL PLOT(X,Y,IPEN,IMARK)
C
C     THIS SUBROUTINE MOVES THE PEN FROM THE CURRENT POSITION TO
C     A NEW POINT.
C
C     THE VARIOUS PARAMETERS IAND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X,Y    - COORDINATES OF THE NEW POINT IN USER'S UNITS.
C        IPEN   - PEN POSITION (0 = PEN UP, NON-ZERO = PEN DOWN).
C        IMARK  - DATA MARK CODE (0-88).
C
C
#ifdef unix
      include 'PLTCOM.DAT'
#else
      include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK THE PLOTTING MODE TO INSURE THAT IT IS SET FOR GRAPH-
C     ICS.
C
      IF(PMODE .EQ. ALPHA)PMODE = VECTOR
C
C     SCALE THE COORDINATES OF THE NEW POINT.
C
      XTEMP = X * XSCALE
      YTEMP = Y * YSCALE
C
C     SET THE PEN POSITION.
C
      PENF = .FALSE.
      IF(IPEN .NE. 0)PENF = .TRUE.
C
C     TO THE POINT.
C
      CALL XYZPLT
C
C     DETERMINE WHETHER A DATA MARK SHOULD BE DRAWN.
C
      IF(IPEN .NE. 0)CALL MARK(IMARK)
      RETURN
C
      END
