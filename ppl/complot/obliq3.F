      SUBROUTINE OBLIQ3(DEGRES)
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
C**    @(#)obliq3.f	1.1    3/10/88
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
C     CALL OBLIQ3(DEGRES)
C
C     THIS SUBROUTINE ESTABLISHES CONTROL INFORMATION FOR OBLIQUE
C     THREE-DIMENSIONAL PROJECTIONS WHERE THE PLANE OF PROJECTION
C     IS PARALLEL TO THE XY-PLANE.
C
C
#ifdef unix
	include 'PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      IF(DEGRES .NE. 0.0)THEN
C
C     SET THE VIEW FLAG.
C
        VIEWF = .TRUE.
C
C     CONVERT THE ANGLE OF PROJECTION TO RADIANS.
C
        TEMP = 0.01745329251994 * DEGRES
C
C     COMPUTE THE ELEMENTS OF THE TRANSFORMATION MATRIX.
C
        TAX = 1.0
        TAY = 0.0
        TAZ = -COS(TEMP)
        TBX = 0.0
        TBY = 1.0
        TBZ = -SIN(TEMP)
        TCX = 0.0
        TCY = 0.0
        TCZ = 0.0
        VDIST = 0.0
      ELSE
C
C     RESET THE VIEW FLAG.
C
        VIEWF = .FALSE.
      ENDIF
      RETURN
C
      END
