	SUBROUTINE ECHO(LINE,ILEN)
C** 
C**    @(#)echo.F	1.3    5/31/88
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
C**		CALL ECHO(LINE,ILEN)
C**
C**	Parameters:
C**		LINE -- Command string to be echoed.
C**		ILEN -- Length of LINE
C**
C**	Description:
C**		Writes LINE on the current ECHO file.  If LINE is too
C**		long a continuation line is also written.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
#ifdef unix
	include 'cmdinc/CMDCOM.INC'
#else
	INCLUDE 'cmdinc:CMDCOM.INC'
#endif
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER LINE*(*),LEV*5
	LOGICAL LEVF
	LEVF=CMDLEV.GT.1
	IF(LEVF)THEN
	    WRITE(LEV,999)CMDLEV
999	    FORMAT('C ',I1,'- ')
	ENDIF
	K=1
200	KS=MIN0(K+69,ILEN)
	IF(KS.NE.ILEN)THEN
	    IF(LEVF)THEN
		WRITE(ECHLUN,998)LEV,LINE(K:KS)
998		FORMAT(A5,A70,'-')
	    ELSE
		WRITE(ECHLUN,996)LINE(K:KS)
996		FORMAT(A70,'-')
	    ENDIF
	ELSE
	    IF(LEVF)THEN
		WRITE(ECHLUN,997)LEV,LINE(K:ILEN)
997		FORMAT(A5,A)
	    ELSE
		WRITE(ECHLUN,995)LINE(K:ILEN)
995		FORMAT(A)
	    ENDIF
	    RETURN
	ENDIF
	K=KS+1
	GOTO 200
	END
