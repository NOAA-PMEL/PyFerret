	SUBROUTINE PARSEX(LINE,ILEN,X,INTIF,M,LABEL,N)
C** 
C**    @(#)parsex.F	1.2    5/26/88
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
C**		CALL PARSEX(LINE,ILEN,X,IF,M,LABEL,N)
C**
C**	Parameters:
C**		LINE -- Line to be parsed   (char*255)
C**		ILEN -- Length of LINE
C**		X -- Numerical array
C**		INTIF -- Flag array
C**			=0 X() not changed
C**			=1 X() replaced
C**		M -- Length of numerical and flag array
C**		LABEL -- Character portion of LINE  (char*255)
C**		N -- Length of LABEL
C**
C**	Description:
C**		Parses the numerical values in LINE placing the results
C**		in X and sets the flag array IF appropriatly.  The
C**		character information starting after the last valid 
C**		number in placed in LABEL.
C**
C**	History:
C**	    Written: 
C**		30-MAY-85	by Donald W. Denbo
C**	    Modified:
C**
C*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER LABEL*(*),LINE*(*),FRMT*30
	DIMENSION X(*),INTIF(*)

*     *jd* variable IF renamed to INTIF for MAC 7.92
*     *jd* 11.94 -  modified for AIX port
*       Modification uses TM_NUMBER to determine if "value" is a number rather
*       than the ERR= branch of a READ which fails on AIX/xlf

        LOGICAL TM_NUMBER                ! ** TMAP mod 11/94 **
C
	LOGICAL FIRST
C
C	Initialize output values
C
	LABEL = ' '
	N=0
	INTIF(1)=0
	M=1
	IF(ILEN.LE.0)THEN
	    M=0
	    RETURN
	ENDIF
	FIRST=.TRUE.
	I=1
C
C	Search for space or comma  (legal seperator)
C
100	IOLD=I
	ISPC=INDEX(LINE(I:),' ')+I-1
	IF(ISPC.EQ.I-1)ISPC=256
	ICMA=INDEX(LINE(I:),',')+I-1
	IF(ICMA.EQ.I-1)ICMA=256
	J=MIN0(ISPC,ICMA,ILEN+1)
C
C	Try to read a number,  if it fails put the rest into
C	LABEL if first is not true
C
	IF(I-J.EQ.0.OR.LINE(I:J-1).EQ.' ')THEN
	    INTIF(M)=0
	ELSE
C
C	Test for LINE(I:I) being +,-,., or 0-9.
C	That is a legal number
C
	    ITST=ICHAR(LINE(I:I))
	    IF(.NOT.(ITST.EQ.43.OR.ITST.EQ.45.OR.ITST.EQ.46
     *		.OR.(ITST.GE.48.AND.ITST.LE.57)))GOTO 1000
	    IWID=J-I
	    if(iwid.eq.1)then
#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(LINE(I:J-1)) ) GOTO 1000  ! TMAP mod 11/94
#endif
		read(line(i:j-1),'(i1)',err=1000)jk
		x(m)=jk
	    else
	    WRITE(FRMT,999)IWID
999	    FORMAT('(F',I3.3,'.0)')
#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(LINE(I:J-1)) ) GOTO 1000  ! TMAP mod 11/94
#endif
	    READ(LINE(I:J-1),FRMT,ERR=1000)X(M)
	    endif
	    INTIF(M)=1
	ENDIF
	FIRST=.FALSE.
	I=J
200	I=I+1
	IF(I.GT.ILEN)RETURN
	IF(LINE(I:I).EQ.' ')GOTO 200
	M=M+1
	GOTO 100
C
C	Error trying to read a number
C	clear the error and put the rest in LABEL
C
1000	continue
#ifndef unix
	CALL ERRTST(64,I)
#endif
	IF(FIRST)THEN
	    FIRST=.FALSE.
	    I=J
	    M=M-1
	    GOTO 200
	ENDIF
	M=M-1
	I=IOLD
	IF(LINE(I:I).EQ.'"')I=I+1
	LABEL=LINE(I:)
	N=ILEN-I+1
	IF(LABEL(N:N).EQ.'"')THEN
	    LABEL(N:N)=' '
	    N=N-1
	ENDIF
	RETURN
	END
