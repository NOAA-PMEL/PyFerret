      SUBROUTINE CIRCLE(XCENT,YCENT,DEG1,DEG2,XRAD,YRAD,PRECSN)
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
C**    @(#)circle.f	1.1    3/10/88
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
C     CALL CIRCLE(XCENT,YCENT,DEG1,DEG2,XRAD,YRAD,PRECSN)
C
C     THIS SUBROUTINE DRAWS AN ARC CENTERED AT A SPECIFIED LOCA-
C     TION AND HAVING A SPECIFIED RADIUS IN THE X- AND Y-DIREC-
C     TIONS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XCENT  - X-COORDINATE OF THE CENTER OF THE ARC IN USER'S
C                 UNITS.
C        YCENT  - Y-COORDINATE OF THE CENTER OF THE ARC IN USER'S
C                 UNITS.
C        DEG1   - INITIAL ANGLE OF THE ARC IN DEGREES RELATIVE TO
C                 THE X-AXIS.
C        DEG2   - FINAL ANGLE OF THE ARC IN DEGREES RELATIVE TO
C                 THE X-AXIS.
C        XRAD   - X-DIRECTION RADIUS OF THE ARC IN USER'S UNITS.
C        YRAD   - Y-DIRECTION RADIUS OF THE ARC IN USER'S UNITS.
C        PRECSN - LENGTH OF THE INDIVIDUAL LINE SEGMENTS IN THE
C                 ARC IN INCHES (OR MM).
C
C
	include 'PLTCOM.DAT'
C
C     CONVERT THE INITIAL AND FINAL ANGLES TO RADIANS.
C
      ANGLE1 = 0.01745329251994 * DEG1
      ANGLE2 = 0.01745329251994 * DEG2
C
C     MOVE TO THE INITIAL POINT ON THE ARC WITH THE PEN UP.
C
      X = XCENT + XRAD * COS(ANGLE1)
      Y = YCENT + YRAD * SIN(ANGLE1)
      CALL PLOT(X,Y,0,0)
      RADIUS = AMAX1(ABS(XRAD * XSCALE),ABS(YRAD * YSCALE))
C
C     IF THE RADIUS IS ZERO, THE ARC CONSISTS OF A SINGLE POINT.
C
      IF(RADIUS .EQ. 0.0 .OR. ANGLE1 .EQ. ANGLE2)GO TO 30
      CHANGE = ABS(PRECSN * FACTOR / RADIUS)
      ANGLE = ANGLE1
      IF(ANGLE1 .GT. ANGLE2)GO TO 20
C
C     DRAW THE COUNTER-CLOCKWISE ARC AS A SERIES OF SMALL LINE
C     SEGMENTS.
C
   10 ANGLE = ANGLE + CHANGE
      IF(ANGLE .GE. ANGLE2)GO TO 30
      X = XCENT + XRAD * COS(ANGLE)
      Y = YCENT + YRAD * SIN(ANGLE)
      CALL PLOT(X,Y,1,0)
      GO TO 10
C
C     DRAW THE CLOCKWISE ARC AS A SERIES OF SMALL LINE SEGMENTS.
C
   20 ANGLE = ANGLE - CHANGE
      IF(ANGLE .LE. ANGLE2)GO TO 30
      X = XCENT + XRAD * COS(ANGLE)
      Y = YCENT + YRAD * SIN(ANGLE)
      CALL PLOT(X,Y,1,0)
      GO TO 20
C
C     MOVE TO THE FINAL POINT ON THE ARC.
C
   30 X = XCENT + XRAD * COS(ANGLE2)
      Y = YCENT + YRAD * SIN(ANGLE2)
      CALL PLOT(X,Y,1,0)
      RETURN
C
      END
