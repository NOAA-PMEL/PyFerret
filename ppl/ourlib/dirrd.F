	SUBROUTINE DIRRD(SELSTR,LUN,FILEC,END)
C** 
C**    @(#)dirrd.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	SUBROUTINE THAT WILL GET MULTIPLE COPIES OF A FILE BASED
C	ON A SELECTION STRING
C
C	D.W. DENBO  MAR 1985
C
C	SELSTR -- 80 CHARACTER STRING CONTAINING WILD CARD FILE NAME
C	LUN  -- INTEGER  (NOT USED)
C	FILEC  -- 80 CHARACTER STRING CONTAINGING FILE NAME
C	END    -- LOGICAL   FALSE NOT DONE  (FILEC CONTAINS NAME)
C			    TRUE IF END OF LIST
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	LOGICAL FIRST,END
	CHARACTER SELSTR*(*),FILEC*(*),FILE*80
	INTEGER*4 STATUS,CONTEXT
	DATA FIRST/.TRUE./
	IF(FIRST)THEN
	  J=INDEX(SELSTR,']')
	  JLN=LNBLK(SELSTR,80)
	  IF(SELSTR(J+1:).EQ.' ')THEN
	    FILE=SELSTR(:J)//'*.*;*'
	  ELSE IF(INDEX(SELSTR,'.').EQ.0)THEN
	    FILE=SELSTR(:JLN)//'.*'
	  ELSE
	    FILE=SELSTR
	  ENDIF
	  CONTEXT=0
	  FIRST=.FALSE.
	ENDIF
	END=.FALSE.

C 	What is this?  Coitenly not unix... *jd* for Linux port
#ifdef unix
	status = 0
#else
	STATUS = LIB$FIND_FILE(FILE,FILEC,CONTEXT)
#endif
***	IF(.NOT.STATUS)THEN
	IF(STATUS .EQ. 0)THEN
	  FIRST=.TRUE.
	  END=.TRUE.
	ENDIF
	RETURN
	END
