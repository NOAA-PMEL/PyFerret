	SUBROUTINE SYMSUB(LINE,ILEN,IER,SYM,IPOS)
C** 
C**    @(#)symsub.f	1.2    5/26/88
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
C**		CALL SYMSUB(LINE,ILEN,IER,SYM)
C**
C**	Parameters:
C**		LINE -- String to have symbols substituted (char*255)
C**		ILEN -- length of LINE before and after substitution
C**		IER -- error code
C**			0 = success
C**			1 = symbol not found
C**			2 = unmatched '
C**		SYM -- Symbol not found
C**		IPOS -- position of error
C**
C**	Description:
C**		Substitutes the values of symbols found in LINE.  Only
C**		symbols of the form 'sym' are substitued.  A double '
C**		is enterpretated as a single '.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER LINE*(*),STR*255,SYM*(*),GET*255
	IER=0
	INXT=1
100	IST=INDEX(LINE(INXT:),'''')+INXT-1
	IF(IST.EQ.INXT-1.OR.IST.GT.ILEN)RETURN
	STR=LINE(:IST-1)
	ISP=INDEX(LINE(IST+1:),'''')+IST
	IF(ISP.EQ.IST)THEN
C	    UNMATCHED '
	    IER=2
	    IPOS=IST
	    RETURN
	ELSE IF((ISP-IST).EQ.1)THEN
C	    DOUBLE '
	    STR(IST:)=LINE(ISP:ILEN)
	    ILEN=ILEN-1
	    INXT=IST+1
	ELSE
C	    EXTRACT SYMBOL AND SUBSTITUTE
	    SYM=LINE(IST+1:ISP-1)
	    CALL GETSYM(SYM,GET,NC,IER)
	    IF(IER.EQ.0)THEN
		STR(IST:)=GET(:NC)
		STR(IST+NC:)=LINE(ISP+1:ILEN)
		ILEN=ILEN-(ISP-IST+1)+NC
		INXT=IST
	    ELSE
		IPOS=IST+1
		RETURN
	    ENDIF
	ENDIF
	LINE=STR(1:ILEN)
	GOTO 100
	END
