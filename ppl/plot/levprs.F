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
*	*jd* MOD 3.29.93 READ fails on SGI with comma in string.  Eliminate.
*	*jd* MOD 11.1.94 READ fails on IBM AIX without it.  Put back in.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	LOGICAL BLNK
	CHARACTER PRSE*(*)

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

#ifdef AIX_XLF
		READ(PRSE(:ISTPP),996,ERR=1000)VAL(I)
#else
		READ(PRSE(:ISTPP-1),996,ERR=1000)VAL(I)
#endif
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
