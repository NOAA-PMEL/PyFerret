	SUBROUTINE LSTSYM(STR,LABEL,NC,ILST,IER)
C** 
C**    @(#)lstsym.F	1.2    5/26/88
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
C**		CALL LSTSYM(STR,LABEL,NC,ILST,IER)
C**
C**	Parameters:
C**		STR -- Symbol name (char*30)
C**		LABEL -- Symbol value (char*255)
C**		NC -- length of LABEL
C**		ILST -- Initially run with ILST = 0
C**			Output is a counter
C**		IER -- error code
C**			0 = success
C**			1 = error or end of file
C**
C**	Description:
C**		Gets the symbol values one by one until all have been
C**		accessed.
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
	INCLUDE 'cmdinc/SYMKEY.INC'
#else
	INCLUDE 'cmdinc:SYMKEY.INC'
#endif
	CHARACTER STR*(*),LABEL*(*)
	IF(ILST.EQ.0)THEN
	    call dbmfirstkey(keyln1,str,ilen,ier)
	else
	    call dbmnextkey(keyln1,str,ilen,ier)
	endif
	if(ier.ne.0)goto 2000
	call dbmfetch(keyln1,str,ilen,label,nc,ier)
	str(ilen+1:)=' '
	ILST=ILST+1
	IER=0
	RETURN
2000	IER=1
	RETURN
	END
