      SUBROUTINE VIEW3(XREF,YREF,ZREF,XDIR,YDIR,ZDIR,DIST)
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
C**    @(#)view3.f	1.1    3/10/88
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
C     CALL VIEW3(XREF,YREF,ZREF,XDIR,YDIR,ZDIR,DIST)
C
C     THIS SUBROUTINE ESTABLISHES CONTROL INFORMATION FOR THREE-
C     DIMENSIONAL PROJECTIONS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XREF   - X-COORDINATE OF THE REFERENCE POINT FOR THE
C                 PROJECTION IN USER'S UNITS.
C        YREF   - Y-COORDINATE OF THE REFERENCE POINT FOR THE
C                 PROJECTION IN USER'S UNITS.
C        ZREF   - Z-COORDINATE OF THE REFERENCE POINT FOR THE
C                 PROJECTION IN USER'S UNITS.
C        XDIR   - X-DIRECTION NUMBER OF THE VIEWING DIRECTION
C                 IN USER'S UNITS.
C        YDIR   - Y-DIRECTION NUMBER OF THE VIEWING DIRECTION
C                 IN USER'S UNITS.
C        ZDIR   - Z-DIRECTION NUMBER OF THE VIEWING DIRECTION
C                 IN USER'S UNITS.
C        DIST   - DISTANCE IN THE VIEWING DIRECTION FROM THE
C                 CENTER OF PROJECTION TO THE REFERENCE POINT
C                 IN INCHES (OR MM).
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      IF(XDIR .NE. 0.0 .OR. YDIR .NE. 0.0 .OR. ZDIR .NE. 0.0)
     * THEN
C
C     SET THE VIEW FLAG.
C
        VIEWF = .TRUE.
C
C     SCALE THE DIRECTION NUMBERS.
C
        XABS = XDIR * XSCALE
        YABS = YDIR * YSCALE
        ZABS = ZDIR * ZSCALE
C
C     COMPUTE THE SINE AND COSINE OF THE ANGLES OF ROTATION.
C
        DISTA = SQRT(XABS * XABS + YABS * YABS + ZABS * ZABS)
        DISTB = SQRT(XABS * XABS + ZABS * ZABS)
        SINA = YABS / DISTA
        COSA = DISTB / DISTA
        IF(DISTB .EQ. 0.0)THEN
          SINB = 0.0
          COSB = 1.0
        ELSE
          SINB = XABS / DISTB
          COSB = ZABS / DISTB
        ENDIF
C
C     COMPUTE THE ELEMENTS OF THE TRANSFORMATION MATRIX.
C
        TAX = COSB
        TAY = 0.0
        TAZ = -SINB
        TBX = -SINA * SINB
        TBY = COSA
        TBZ = -SINA * COSB
        TCX = COSA * SINB
        TCY = SINA
        TCZ = COSA * COSB
C
C     SCALE THE REFERENCE POINT.
C
        XABS = XREF * XSCALE
        YABS = YREF * YSCALE
        ZABS = ZREF * ZSCALE
C
C     TRANSFORM THE REFERENCE POINT.
C
        AREF = XABS * TAX + YABS * TAY + ZABS * TAZ
        BREF = XABS * TBX + YABS * TBY + ZABS * TBZ
        CREF = XABS * TCX + YABS * TCY + ZABS * TCZ
C
C     CONVERT THE VIEW DISTANCE TO INTERNAL INCREMENTS.
C
        VDIST = FACTOR * DIST
      ELSE
C
C     RESET THE VIEW FLAG.
C
        VIEWF = .FALSE.
      ENDIF
      RETURN
C
      END
