      SUBROUTINE BELL
C** 
C**    @(#)bell.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	include 'pplinc/PLTCOM.DAT'	
#else
	include 'pplinc:PLTCOM.DAT'	
#endif
C
C     CALL BELL
C
C     THIS SUBROUTINE RINGS THE BELL.
C
C
C
C     RING THE BELL.
C
      CALL CHOUT(BEL,1)
C
C     WRITE OUT THE CHARACTER BUFFER.
C
      CALL CHDMP
      RETURN
C
      END
