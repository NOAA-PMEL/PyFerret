	SUBROUTINE SYMCNV(SYM,SYS)
C** 
C**    @(#)symcnv.f	1.2    5/26/88
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
C**		CALL SYMCNV(SYM,SYS)
C**
C**	Parameters:
C**		SYM -- Symbol name (char*30)
C**		SYS -- Symbol is SYSTEM type (logical)
C**
C**	Description:
C**		Converts the symbol name to upper case and evaluates the
C**		form SYM(#) to SYM#.
C**
C**	History:
C**	    Written:
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**

*       *jd* 11.94 - modified for AIX port
*       Modification uses TM_NUMBER to determine if "value" is a number rather
*       than the ERR= branch of a READ which fails on AIX/xlf
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

        LOGICAL TM_NUMBER      

	LOGICAL SYS,SYS2
	CHARACTER SYM*(*),FRMT*30,SYM2*30,VALUE*255
	SYS=.FALSE.
	IF(SYM(1:1).EQ.'*')THEN
	    SYS=.TRUE.
	    FRMT=SYM(2:)
	    SYM=FRMT
	ENDIF
200	I=INDEX(SYM,'(')+1
	J=INDEX(SYM,')')-1
	CALL UPPER(SYM,30)
	IF(I.EQ.1.OR.J.EQ.-1)RETURN
	if(i.eq.j)then

#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(sym(i:j)) ) GOTO 100  ! TMAP mod
#endif
	    read(sym(i:j),'(i1)',err=100)jk
	    frmt=sym(i:j)
	else
	WRITE(FRMT,999)J-I+1
999	FORMAT('(E',I2.2,'.0)')
#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(sym(i:j)) ) GOTO 100  ! TMAP mod
#endif
	READ(SYM(I:J),FRMT,ERR=100)X
	WRITE(FRMT,998)IFIX(X+0.5)
998	FORMAT(I2.2)
	endif
	IF(FRMT(1:1).EQ.'0')THEN
	    SYM(I-1:)=FRMT(2:)
	ELSE
	    SYM(I-1:)=FRMT
	ENDIF
	RETURN
100	SYM2=SYM(I:J)
	ILEN=J-I+1
	CALL CMLJST(SYM2,ILEN)
	SYS2=SYM2(1:1).EQ.'*'
	IF(SYS2)THEN
	    FRMT=SYM2(2:)
	    SYM2=FRMT
	ENDIF
	CALL GTSYM2(SYM2,VALUE,NC,SYS2,IER)
	IF(IER.NE.0)THEN
	    SYM(I:)=' '
	    RETURN
	ELSE
	    SYM(I:)=VALUE(:NC)
	    SYM(I+NC:)=')'
	ENDIF
	GOTO 200
	END


