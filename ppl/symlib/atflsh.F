	SUBROUTINE ATFLSH
C**
C**    @(#)atflsh.F	1.2    5/26/88
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL ATFLSH
C**
C**	Parameters:
C**
C**	Description:
C**		ATFLSH deletes all temporary user key files and returns to
C**		the lowest level of command files.
C**
C**	History:
C**	    Written: 
C**		26-JAN-86	by Donald W. Denbo
C**	    Modified:
C**
C**
#ifdef unix
	include 'cmdinc/CMDCOM.INC'
#else
	INCLUDE 'cmdinc:CMDCOM.INC'
#endif
C
C	CALL ATEND REPEATEDLY
C
	DO 100 I=CMDLEV,2,-1
	CALL ATEND
100	CONTINUE
	RETURN
	END
