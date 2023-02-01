      SUBROUTINE TFORMI(X,Y,Z)
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
C**    @(#)tformi.f	1.1    3/10/88
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
C     CALL TFORMI(X,Y,Z)
C
C     THIS SUBROUTINE TRANSFORMS THE COORDINATES OF A POINT
C     FROM THE SCREEN COORDINATE SYSTEM TO THE USER'S COOR-
C     DINATE SYSTEM.  PERSPECTIVE IS REMOVED IF APPROPRIATE.
C
C
	include 'PLTCOM.DAT'
      REAL*8 PERSPC
C
      AROT = AOLD
      BROT = BOLD
C
C     CHECK THE MIRROR-IMAGE FLAGS.
C
      IF(MIRAF)AROT = ASIZE - AROT
      IF(MIRBF)BROT = BSIZE - BROT
C
C     CHECK THE VIEW FLAG.
C
      IF(VIEWF)THEN
C
C     CHECK THE ROTATION FLAG.
C
        IF(ROTAT)THEN
C
C     REVERSE TRANSLATE AND REVERSE ROTATE THE POINT.
C
          AROT = AROT - AORIGN
          BROT = BROT - BORIGN
          ATEMP = AROT * ROTCOS + BROT * ROTSIN
          BTEMP = -AROT * ROTSIN + BROT * ROTCOS
        ELSE
C
C     REVERSE TRANSLATE BUT DO NOT REVERSE ROTATE THE POINT.
C
          ATEMP = AROT - AORIGN
          BTEMP = BROT - BORIGN
        ENDIF
C
C     IF APPROPRIATE, REMOVE PERSPECTIVE FROM THE POINT.
C
        IF(VDIST .NE. 0.0)THEN
          PERSPC = (VDIST - COLD) / VDIST
          ATEMP = ATEMP * PERSPC
          BTEMP = BTEMP * PERSPC
        ENDIF
C
C     REVERSE TRANSFORM THE POINT.
C
        ATEMP = ATEMP + AREF
        BTEMP = BTEMP + BREF
        CTEMP = COLD + CREF
        X = ATEMP * TAX + BTEMP * TBX + CTEMP * TCX
        Y = ATEMP * TAY + BTEMP * TBY + CTEMP * TCY
        Z = ATEMP * TAZ + BTEMP * TBZ + CTEMP * TCZ
      ELSE
C
C     CHECK THE ROTATION FLAG.
C
        IF(ROTAT)THEN
C
C     REVERSE TRANSLATE AND REVERSE ROTATE THE POINT.
C
          AROT = AROT - AORIGN
          BROT = BROT - BORIGN
          X = AROT * ROTCOS + BROT * ROTSIN
          Y = -AROT * ROTSIN + BROT * ROTCOS
        ELSE
C
C     REVERSE TRANSLATE BUT DO NOT REVERSE ROTATE THE POINT.
C
          X = AROT - AORIGN
          Y = BROT - BORIGN
        ENDIF
      ENDIF
      RETURN
C
      END
