      SUBROUTINE TFORM(X,Y,Z)
C** 
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
#ifdef unix
      include 'pplinc/PLTCOM.DAT'
#else
      include 'pplinc:PLTCOM.DAT'
#endif
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
