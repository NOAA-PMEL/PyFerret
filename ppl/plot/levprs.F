	SUBROUTINE LEVPRS(PRSE,VAL,INTIF,IER,IPOS)
C** 
C**    @(#)levprs.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	LOGICAL BLNK
	CHARACTER PRSE*255

*       variable IF renamed to INTIF for MAC *jd* 7.92

	DIMENSION VAL(4),INTIF(4)
	IER=0
	IPOS=1
	I=1
30	ICM=INDEX(PRSE,',')
	IF(ICM.EQ.0)ICM=256
	IBL=INDEX(PRSE,' ')
	IF(IBL.EQ.0)IBL=256
	ISTPP=MIN0(ICM,IBL,255)
	BLNK=PRSE(ISTPP:ISTPP).EQ.' '
	IF(ISTPP.NE.1)THEN
		INTIF(I)=1
		PRSE(ISTPP:ISTPP)=','
		READ(PRSE(:ISTPP),996,ERR=1000)VAL(I)
996		FORMAT(E20.0)
	ENDIF
20	ISTPP=ISTPP+1
	IF(ISTPP.GT.255)GOTO 40
	IF(PRSE(ISTPP:ISTPP).EQ.' '.OR.
     *	(BLNK.AND.PRSE(ISTPP:ISTPP).EQ.','))GOTO 20
	PRSE=PRSE(ISTPP:)
	IPOS=IPOS+ISTPP-1
	I=I+1
	IF(I.LE.4)GOTO 30
40	RETURN
1000	IER=9
	RETURN
	END
