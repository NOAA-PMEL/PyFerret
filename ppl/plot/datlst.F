      SUBROUTINE DATLST(LLUN,X,NSIZE,NXS,NYS) 
C** 
C**    @(#)datlst.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	DIMENSION X(NSIZE)
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/LINES.INC'
	INCLUDE 'pplinc/HD.INC'
	INCLUDE 'pplinc/CMRD.INC'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:LINES.INC'
	INCLUDE 'pplinc:HD.INC'
	INCLUDE 'pplinc:CMRD.INC'
#endif
	IY=NSIZE/2
	NXL=1 
	NYL=1 
	IF(M.GE.1.AND.IFLG(1).EQ.1)NXL=IFIX(P(1)) 
	IF(M.GE.3.AND.IFLG(3).EQ.1)NYL=IFIX(P(3)) 
	IF(JTYPE.LE.0.OR.JTYPE.EQ.2)THEN 
	    NXH=NX 
	    NYH=NY 
	    IF(M.GE.2.AND.IFLG(2).EQ.1)NXH=IFIX(P(2)) 
	    IF(M.GE.4.AND.IFLG(4).EQ.1)NYH=IFIX(P(4)) 
	    IF(JTYPE.EQ.-1.OR.JTYPE.EQ.2)THEN
		IF(M.GE.5.AND.IFLG(5).EQ.1.AND.P(5).EQ.2)THEN
		    WRITE(LLUN,9994)'SECOND'
9994		    FORMAT(9X,A6,' COMPONENT')
		    IVLEN=NX*NY
 	ELSE
		    WRITE(LLUN,9994)'FIRST'
		    IVLEN=0
		ENDIF
	    ELSE
		IVLEN=0
	    ENDIF
	    K=(NXH-NXL)/7+1 
	    DO 100 K1=1,K 
	    K8=(K1-1)*7+(NXL-1) 
	    IM=NXH-K8 
	    IF(IM.GT.7)IM=7 
	    WRITE(LLUN,9999)(I+K8,I=1,IM) 
9999	    FORMAT(9X,7I10) 
	    DO 101 J=NYL,NYH 
	    WRITE(LLUN,9998)J,(X(IVLEN+NXS*(J-1)+I+K8),I=1,IM) 
101	    CONTINUE
9998	    FORMAT(6X,I6,1P7E10.2) 
	    WRITE(LLUN,'(/)') 
100	    CONTINUE
	ELSE 
	    IBASE=1 
	    NXH=32000 
	    NYH=LNUM 
	    IF(M.GE.2.AND.IFLG(2).EQ.1)NXH=IFIX(P(2)) 
	    IF(M.GE.4.AND.IFLG(4).EQ.1)NYH=IFIX(P(4)) 
	    DO 300 J=1,NYL-1 
300	    IBASE=IBASE+LLENG(J) 
	    DO 200 J=NYL,NYH 
	    WRITE(LLUN,9997)J,LLENG(J) 
9997	    FORMAT(//9X,'LINE=',I4,'  LENGTH=',I6) 
	    WRITE(LLUN,9996) 
9996	    FORMAT(11X,'I',10X,'X',15X,'Y') 
	    IB=IBASE+LLENG(J)-1 
	    IF(IB.GT.IBASE+NXH-1)IB=IBASE+NXH-1 
	    WRITE(LLUN,9995)(I-IBASE+1,X(I),X(IY+I),I=IBASE+NXL-1,IB) 
9995	    FORMAT(6X,I6,1P2E15.4) 
	    IBASE=IBASE+LLENG(J) 
200	    CONTINUE 
	END IF 
	RETURN
	END 
