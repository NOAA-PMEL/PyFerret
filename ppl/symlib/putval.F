	SUBROUTINE PUTVAL(STR,X,IPRES,IER)
C** 
C**    @(#)putval.f	1.2    5/26/88
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
C**		CALL PUTVAL(STR,X,IPRES,IER)
C**
C**	Parameters:
C**		STR -- Symbol name (char*30)
C**		X -- Real value of symbol
C**		IPRES -- number of digits, i.e, 1PG10.IPRES
C**		IER -- error code
C**			0 = sucess
C**			1 = error other than symbol not found
C**
C**	Description:
C**		Updates or creates the value for symbol STR.
C**
C**	History:
C**	    Written:
C**		14-JAN-1987	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STR*(*),VAL*255,FRMT*30
	IP=IPRES
	IF(IPRES.GT.10.OR.IPRES.LT.0)IP=4
	IL=IP+7
	WRITE(FRMT,999)IL,IP
999	FORMAT('(1PG',I3.3,'.',I3.3,')')
	WRITE(VAL,FRMT)X
100	IF(VAL(1:1).EQ.' ')THEN
	    VAL=VAL(2:)
	    IL=IL-1
	    GOTO 100
	ENDIF
	NC=LNBLK(VAL,IL)
	CALL PUTSYM(STR,VAL,NC,IER)
	RETURN
	END
