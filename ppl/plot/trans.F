      SUBROUTINE TRANS(I,X,Y,XT,YT) 
C** 
C**    @(#)trans.F	1.1    3/10/88
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
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/LINES.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:LINES.INC'
#endif
      XT=X*XFCT(I)+XOFF(I) 
      YT=Y*YFCT(I)+YOFF(I) 
      IF(ITYPEX.NE.1)XT=ALOG10(ABS(XT)) 
      IF(ITYPEY.NE.1)YT=ALOG10(ABS(YT)) 
      RETURN 
      END 
