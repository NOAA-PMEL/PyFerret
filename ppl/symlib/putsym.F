	SUBROUTINE PUTSYM(STR,LABEL,NC,IER)
C** 
C**    @(#)putsym.F	1.2    5/26/88
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
C**		CALL PUTSYM(STR,LABEL,NC,IER)
C**
C**	Parameters:
C**		STR -- Symbol name (char*30)
C**		LABEL -- Value of symbol (char*255)
C**		NC -- length of LABEL
C**		IER -- error code
C**			0 = sucess
C**			1 = error other than symbol not found
C**
C**	Description:
C**		Updates or creates the value for symbol STR.
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
	INTEGER LUN
	CHARACTER STR*(*),LABEL*(*)
	CALL SYMCNV(STR,SYS)
	IER=0
	LUN=KEYLN1
	ilen=lnblk(str,30)
	IF(SYS)GOTO 100
	IF(SMKEY2)LUN=KEYLN2
100	call dbmstore(lun,str,ilen,label,nc,ier)
	RETURN
	END
