	SUBROUTINE STPRMP(PMT,N)
C** 
C**    @(#)stprmp.F	1.2    5/26/88
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
C**		CALL STPRMP(PMT,N)
C**
C**	Parameters:
C**		PMT -- Prompt to be used when input is from a terminal
C**			(char*30)
C**		N -- Lenght of the prompt
C**
C**	Description:
C**		Sets the prompt for the GETLIN subroutine.
C**
C**	History:
C**	    Written: 
C**		21-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	INCLUDE 'cmdinc/CMDCOM.INC'
	CHARACTER PMT*(*)
	IF(N.NE.0)THEN
	    PRMPT=PMT
	    INP=N
	ELSE
	    PRMPT='*'
	    INP=1
	ENDIF
#else
	INCLUDE 'cmdinc:CMDCOM.INC'
        CHARACTER PMT*30,FRMT*12,LF*1
        BYTE CRB,LFB
        EQUIVALENCE (LFB,LF)
        DATA LFB/10/
        IF(N.NE.0)THEN
            PRMPT=LF
            PRMPT(2:)=PMT
            INP=N+1
        ELSE
            PRMPT=LF
            PRMPT(2:2)='*'
            INP=2
        ENDIF
#endif
	RETURN
	END
