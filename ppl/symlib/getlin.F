	SUBROUTINE GETLIN(LINE,ILEN)
C** 
C**    @(#)getlin.F	1.2    5/26/88
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
C**		CALL GETLIN(LINE,ILEN)
C**
C**	Parameters:
C**		LINE -- Line from command input device   (char*255)
C**		ILEN -- Length of LINE
C**
C**	Description:
C**		Gets the next line from the command input device.
C**
C**	History:
C**	    Written: 
C**		21-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
C**
#ifdef unix
	INCLUDE 'cmdinc/CMDCOM.INC'
	INCLUDE 'cmdinc/SYSTEM.INC'
#else
	INCLUDE 'cmdinc:CMDCOM.INC'
	INCLUDE 'cmdinc:SYSTEM.INC'
#endif
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER PMPT*30,LINE*(*),IN*255
	IP=INP
	PMPT=PRMPT
	K=1
100	IN=' '
#ifdef unix
	IF(CMDFIL(1:8).EQ.'/dev/tty')THEN
#else
	if(cmdfil(1:3).eq.'TT:')then
#endif
	    CALL CHARIN(IN,255,PMPT,IP)
	ELSE
	    READ(CMDLUN,999,END=901)IN
999	    FORMAT(A)
	    LINCNT=LINCNT+1
	ENDIF
	ILEN=LNBLK(IN,255)
	LINE(K:K+ILEN-1)=IN(1:ILEN)
	IF(IN(ILEN:ILEN).EQ.'-')THEN
	    K=K+ILEN-1
C
C	SINCE LF IS FIRST CHARACTER
C
	    PMPT=PRMPT
	    PMPT(2:2)='_'
	    PMPT(3:)=PRMPT(2:)
	    IP=INP+1
	    GOTO 100
	ENDIF
	ILEN=K+ILEN-1
	RETURN
C
C	EOF READ GO UP ONE LEVEL IN COMMAND FILE STRUCTURE
C
901	CALL ATEND
	IF(DONEF)RETURN
	GOTO 100
	END
