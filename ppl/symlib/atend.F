	SUBROUTINE ATEND
C** 
C**    @(#)atend.F	1.2    5/26/88
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
C**		CALL ATEND
C**
C**	Parameters:
C**
C**	Description:
C**		ATEND closes the currently open command file and user
C**		key file and returns to the previous command file.
C**
C**	History:
C**	    Written: 
C**		26-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		15-JUN-86	by Donald W. Denbo
C**		  changes for memory buffer added
C**
C**
#ifdef unix
	include 'cmdinc/SYSTEM.INC'
	include 'cmdinc/CMDCOM.INC'
	include 'cmdinc/SYMKEY.INC'
	INCLUDE 'pplinc/LUNITS.INC'
#else
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:CMDCOM.INC'
	INCLUDE 'cmdinc:SYMKEY.INC'
	include 'pplinc:lunits.inc'
#endif
	CHARACTER KFILE*30,SYM*30
	DONEF=CMDLEV.EQ.1
	IF(DONEF)THEN
C
C	DONE -- FINISH UP AND EXIT
C
	    RETURN
	ENDIF
C
C	CLOSE USER KEY FILE AND CURRENT COMMAND FILE
C
	call dbmclear(keyln2)
	keyln2=keyln2-1
	CLOSE(CMDLUN)
C
C	RESET COMMAND FILE 
C
	CMDLEV=CMDLEV-1
	IFLEV=CMIFLV(CMDLEV)
	WHLEV=CMWHLV(CMDLEV)
	SKIPIF=.FALSE.
	SKIPWH=.FALSE.
	DO 10 I=1,WHLEV
10	WHLINE(I)=CMWHLN(CMDLEV,I)
	CMDFIL=CMFILE(CMDLEV)
	LINCNT=CMLINE(CMDLEV)
	ECHOF=CMECHO(CMDLEV)
	DEBUGF=CMDEBG(CMDLEV)
	QUIETF=CMQUIE(CMDLEV)
	LOGCMF=CMLOG(CMDLEV)
#ifdef unix
	IF(CMDFIL.EQ.'/dev/tty')THEN
#else
	if(cmdfil.eq.'TT:')then
#endif
	    TERMF=.TRUE.
	    IF(.NOT.QUIETF)WRITE(LTTOUT,997)
997	    FORMAT(' Control returned to keyboard')
#ifdef unix
	    OPEN(CMDLUN,FILE=CMDFIL,STATUS='OLD')
#endif
	ELSE IF(CMDFIL.EQ.'$$MEMBUF$$')THEN
	    TERMF=.FALSE.
	    MEMBUF=.TRUE.
	ELSE
C
C	OPEN OLD COMMAND FILE AND SKIP TO CORRECT LINE
C
#ifdef unix
	    OPEN(CMDLUN,FILE=CMDFIL,STATUS='OLD')
#else
	    open(cmdlun,file=cmdfil,readonly,status='old')
#endif
	    DO 100 I=1,LINCNT
	    READ(CMDLUN,998)
998	    FORMAT(1X)
100	    CONTINUE
	ENDIF
	ISYM=LNBLK(CMDFIL,80)
	SYM='*PPL$COMMAND_FILE'
	CALL PUTSYM(SYM,CMDFIL,ISYM,IER)
	IF(CMDLEV.GT.1)THEN
C
C	OPEN OLD USER KEY FILE
C
	    WRITE(KFILE,999)CMDLEV
999	    FORMAT('PPL$KEY.',I3.3)
	    call dbmopen(kfile,11,keyln2)
	ELSE
	    SMKEY2=.FALSE.
	ENDIF
	RETURN
	END
