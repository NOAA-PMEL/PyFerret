	SUBROUTINE GETSYM(STR,LABEL,NC,IER)
C** 
C**    @(#)getsym.F	1.3    5/31/88
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
C**		CALL GETSYM(STR,LABEL,NC,IER)
C**
C**	Parameters:
C**		STR -- 30 CHARACTER SYMBOL  (char*30)
C**		LABEL -- Value of symbol (char*255)
C**		NC -- length of LABEL
C**		IER -- error code
C**			0 = success
C**			1 = symbol not found
C**
C**	Description:
C**		Gets value for symbol.
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
	CHARACTER STR*(*),LABEL*(*),DTE*24
#else
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:SYMKEY.INC'
	character str*30,label*255,dummy*30
#endif
	LOGICAL SYS
	CALL SYMCNV(STR,SYS)
	IER=0
	LABEL=' '
C
C	Check for special symbols
C
	IF(STR.EQ.'TIME')THEN
#ifdef unix
#ifdef F90_DATE_TIME
	    CALL FTIME(dte)
	    label=dte(1:8)
	    NC=8
	
#else
	    CALL fdate(dte)
	    label=dte(12:19)
	    NC=9
#endif
#else
	    call time(label)
	    NC=8
#endif
	ELSE IF(STR.EQ.'DATE')THEN
#ifdef unix
	    CALL FDATE(dte)
#ifdef F90_DATE_TIME
	    label=dte(4:6)//' '//dte(1:2)//' '//dte(8:9)
	    NC=9
#else
	    label=dte(5:11)//dte(21:24)
	    NC=11
#endif
#else
	    call date(label)
	    nc=9
#endif
	ELSE
C
C	Otherwise get symbol value from file
C
	    CALL GTSYM2(STR,LABEL,NC,SYS,IER)
	ENDIF
	RETURN
	END
	SUBROUTINE GTSYM2(STR,LABEL,NC,SYS,IER)
#ifdef unix
	INCLUDE 'cmdinc/SYSTEM.INC'
	INCLUDE 'cmdinc/SYMKEY.INC'
#else
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:SYMKEY.INC'
#endif
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STR*(*),LABEL*(*),dummy*30
	LOGICAL SYS
	ilen=lnblk(str,30)
	IF(SYS)GOTO 100
	IF(SMKEY2)THEN
	    call dbmfetch(keyln2,str,ilen,label,nc,ier)
	    if(ier.ne.0)goto 100
	    RETURN
	ENDIF
100	call dbmfetch(keyln1,str,ilen,label,nc,ier)
	RETURN
	END
