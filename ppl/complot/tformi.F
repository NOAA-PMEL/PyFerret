      SUBROUTINE TFORMI(X,Y,Z)
C** 
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
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
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
