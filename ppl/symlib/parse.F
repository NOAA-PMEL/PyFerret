	SUBROUTINE PARSE(LINE,ILEN,ICOM,LAB,N,IST)
C** 
C**    @(#)parse.f	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL PARSE(LINE,ILEN,ICOM,LAB,N)
C**
C**	Parameters:
C**		LINE -- Line to be parsed   (char*255)
C**		ILEN -- Length of ILEN
C**		ICOM -- Command from LINE   (char*255)
C**		LAB -- rest of LINE  (char*255)
C**		N -- Length of LAB
C**		IST -- Start position of LAB
C**
C**	Description:
C**		Remove the command from LINE and places the rest of LINE
C**		in LAB.
C**
C**	History:
C**	    Written: 
C**		22-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
C*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER ICOM*(*),LAB*(*),LINE*(*)
	ICOM=' '
	LAB=' '
C
C	Search for the fist space or comma
C
	N=0
	IST=0
	ISPC=INDEX(LINE,' ')
	IF(ISPC.EQ.0)ISPC=256
	ICMA=INDEX(LINE,',')
	IF(ICMA.EQ.0)ICMA=256
	I=MIN0(ISPC,ICMA,256,ILEN+1)
	ICOM=LINE(:I-1)
	CALL UPPER(ICOM,255)
100	I=I+1
	IF(I.GT.ILEN)RETURN
	IF(LINE(I:I).EQ.' ')GOTO 100
	IF(LINE(I:I).EQ.'"')I=I+1
	LAB=LINE(I:)
	IST=I
	N=ILEN-I+1
	IF(LAB(N:N).EQ.'"')THEN
	    LAB(N:N)=' '
	    N=N-1
	ENDIF
	RETURN
	END
