	SUBROUTINE CMLJST(LINE,ILEN)
C** 
C**    @(#)cmljst.f	1.2    5/26/88
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
C**		CALL CMLJST(LINE,ILEN)
C**
C**	Parameters:
C**		line -- input string to be left jusified
C**		ilen -- length of input string on input and output
C**
C**	Description:
C**		Removes blank spaces and tabs from LINE
C**
C**	History:
C**	    Written: 
C**		6-SEP-86	by Donald W. Denbo
C**	    Modified:
C**
C** 
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER LINE*(*)
	INTEGER ILEN
	IF(LINE(:ILEN).EQ.' ')RETURN
100	J=ICHAR(LINE(1:1))
	IF(J.EQ.9.OR.J.EQ.32)THEN
	    ILEN=ILEN-1
	    LINE=LINE(2:)
	    GOTO 100
	ENDIF
	RETURN
	END
