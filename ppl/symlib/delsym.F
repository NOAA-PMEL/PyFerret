	SUBROUTINE DELSYM(SYM,IER)
C** 
C**    @(#)delsym.F	1.2    5/26/88
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
C**		CALL DELSYM(SYM,IER)
C**
C**	Parameters:
C**		SYM -- Symbol name (char*30)
C**		IER -- error code
C**			0 = sucess
C**			1 = symbol not found
C**
C**	Description:
C**		Deletes symbol from table.  Does not recover space.
C**
C**	History:
C**	    Written:
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	INCLUDE 'cmdinc/SYSTEM.INC'
	INCLUDE 'cmdinc/SYMKEY.INC'
#else
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:SYMKEY.INC'
#endif
	LOGICAL SYS
	CHARACTER SYM*(*)
	CALL SYMCNV(SYM,SYS)
	IER=0
	IF(SYS)GOTO 100
	IF(SMKEY2)THEN
	    ilen=lnblk(sym,30)
	    call dbmdelete(keyln2,sym,ilen,ier)
	    RETURN
	ENDIF
100	ilen=lnblk(sym,30)
	call dbmdelete(keyln1,sym,ilen,ier)
	RETURN
	END
