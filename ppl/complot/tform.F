      SUBROUTINE TFORM(X,Y,Z)
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
C**    @(#)tform.f	1.1    3/10/88
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
C     CALL TFORM(X,Y,Z)
C
C     THIS SUBROUTINE TRANSFORMS THE COORDINATES OF A POINT
C     FROM THE USER'S COORDINATE SYSTEM TO THE SCREEN COOR-
C     DINATE SYSTEM.  PERSPECTIVE IS APPLIED IF APPROPRIATE.
C
C
      include 'PLTCOM.DAT'
      REAL*8 PERSPC
C
C     CHECK THE VIEW FLAG.
C
      IF(VIEWF)THEN
C
C     TRANSFORM THE POINT.
C
        ATEMP = X * TAX + Y * TAY + Z * TAZ - AREF
        BTEMP = X * TBX + Y * TBY + Z * TBZ - BREF
        COLD = X * TCX + Y * TCY + Z * TCZ - CREF
C
C     IF APPROPRIATE, APPLY PERSPECTIVE TO THE POINT.
C
        IF(VDIST .NE. 0.0)THEN
          PERSPC = (VDIST - COLD) / VDIST
          IF(PERSPC .EQ. 0.0)THEN
            ATEMP = 0.0
            BTEMP = 0.0
          ELSE
            ATEMP = ATEMP / PERSPC
            BTEMP = BTEMP / PERSPC
          ENDIF
        ENDIF
C
C     CHECK THE ROTATION FLAG.
C
        IF(ROTAT)THEN
C
C     ROTATE AND TRANSLATE THE POINT.
C
          ANEW = ATEMP * ROTCOS - BTEMP * ROTSIN + AORIGN
          BNEW = ATEMP * ROTSIN + BTEMP * ROTCOS + BORIGN
        ELSE
C
C     TRANSLATE BUT DO NOT ROTATE THE POINT.
C
          ANEW = ATEMP + AORIGN
          BNEW = BTEMP + BORIGN
        ENDIF
      ELSE
C
C     CHECK THE ROTATION FLAG.
C
        IF(ROTAT)THEN
C
C     ROTATE AND TRANSLATE THE POINT.
C
          ANEW = X * ROTCOS - Y * ROTSIN + AORIGN
          BNEW = X * ROTSIN + Y * ROTCOS + BORIGN
        ELSE
C
C     TRANSLATE BUT DO NOT ROTATE THE POINT.
C
          ANEW = X + AORIGN
          BNEW = Y + BORIGN
        ENDIF
      ENDIF
C
C     CHECK THE MIRROR-IMAGE FLAGS.
C
      IF(MIRAF)ANEW = ASIZE - ANEW
      IF(MIRBF)BNEW = BSIZE - BNEW
      RETURN
C
      END
