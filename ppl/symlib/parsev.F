	SUBROUTINE PARSEV(LINE,UPLINE,ILEN,RESULT,IER,IPOS)
C** 
C**    @(#)parsev.f	1.2    5/26/88
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
C**		call parsev(line,upline,ilen,result,ier,ipos)
C**
C**	Parameters:
C**		line -- input line (char*255)
C**		upline -- input line capitals (char*255)
C**		ilen -- length of line
C** 	result -- output (logical)
C**		ier -- error code
C**		       6 = no operator found
C**		ipos -- position error was found in line
C**
C**	Description:
C**		Parses the logical expression for both if and while
C**		statements.  Returns a logical result, either true or
C**		false.
C**
C**	History:
C**	    Written: 
C**		3-SEP-86	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER LINE*(*),UPLINE*(*),EXP1*255,EXP2*255
	LOGICAL RESULT,NUM1,NUM2,NUM
	IER=0
C
C	CHECK FOR OPERATORS
C
	IEQ=INDEX(UPLINE,'.EQ.')
	IF(IEQ.LE.0)THEN
	    IEQ=256
	ELSE
	    ISW=1
	ENDIF
	INE=INDEX(UPLINE,'.NE.')
	IF(INE.LE.0)THEN
	    INE=256
	ELSE
	    ISW=2
	ENDIF
	ILT=INDEX(UPLINE,'.LT.')
	IF(ILT.LE.0)THEN
	    ILT=256
	ELSE
	    ISW=3
	ENDIF
	IGT=INDEX(UPLINE,'.GT.')
	IF(IGT.LE.0)THEN
	    IGT=256
	ELSE
	    ISW=4
	ENDIF
	ILE=INDEX(UPLINE,'.LE.')
	IF(ILE.LE.0)THEN
	    ILE=256
	ELSE
	    ISW=5
	ENDIF
	IGE=INDEX(UPLINE,'.GE.')
	IF(IGE.LE.0)THEN
	    IGE=256
	ELSE
	    ISW=6
	ENDIF
	IOP=MIN0(IEQ,INE,ILT,IGT,ILE,IGE)
	IF(IOP.GE.256)THEN
C
C	NO OPERATOR FOUND
C
	    IER=6
	    IPOS=1
	    RETURN
	ENDIF
C
C	GET EXPRESSIONS
C
	I=1
100	IF(LINE(I:I).NE.' ')GOTO 110
	I=I+1
	GOTO 100
110	EXP1=LINE(I:IOP-1)
	LEN1=LNBLK(EXP1,IOP-I)
	I=IOP+4
120	IF(LINE(I:I).NE.' ')GOTO 130
	I=I+1
	GOTO 120
130	EXP2=LINE(I:ILEN)
	LEN2=LNBLK(EXP2,ILEN-I+1)
C
C	EVALUATE EXPRESSIONS
C
	CALL EXPEVL(EXP1,LEN1,X1,NUM1,IER)
	CALL EXPEVL(EXP2,LEN2,X2,NUM2,IER)
	NUM=NUM1.AND.NUM2
C
C	DO TEST
C
	IF(NUM)THEN
	    GOTO(201,202,203,204,205,206)ISW
201	    RESULT=X1.EQ.X2
     	    GOTO 210
202	    RESULT=X1.NE.X2
	    GOTO 210
203	    RESULT=X1.LT.X2
	    GOTO 210
204	    RESULT=X1.GT.X2
	    GOTO 210
205	    RESULT=X1.LE.X2
	    GOTO 210
206	    RESULT=X1.GE.X2
210	    CONTINUE
	ELSE
	    GOTO(301,302,303,304,305,306)ISW
301	    RESULT=EXP1(:LEN1).EQ.EXP2(:LEN2)
	    GOTO 310
302	    RESULT=EXP1(:LEN1).NE.EXP2(:LEN2)
	    GOTO 310
303	    RESULT=EXP1(:LEN1).LT.EXP2(:LEN2)
	    GOTO 310
304	    RESULT=EXP1(:LEN1).GT.EXP2(:LEN2)
	    GOTO 310
305	    RESULT=EXP1(:LEN1).LE.EXP2(:LEN2)
	    GOTO 310
306	    RESULT=EXP1(:LEN1).GE.EXP2(:LEN2)
310	    CONTINUE
	ENDIF
	RETURN
	END
	SUBROUTINE EXPEVL(STRING,SLEN,X,NUM,IER)
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
C**		call expevl(string,slen,x,num,ier)
C**
C**	Parameters:
C**		string -- input expression (char*255)
C**		slen  -- length of string
C**		x -- numeric result
C**		num -- logical for number (logical)
C**		ier -- error code
C**		       7 = unmatched "
C**
C**	Description:
C**		Evaluates expression.
C**
C**	History:
C**	    Written: 
C**		3-SEP-86	by Donald W. Denbo
C**	    Modified:
C**
*       *jd* 11.94 - modified for AIX port
*       Modification uses TM_NUMBER to determine if "value" is a number rather
*       than the ERR= branch of a READ which fails on AIX/xlf

C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STRING*(*),TEMP*255,FRMT*30
	INTEGER SLEN
	LOGICAL NUM

        LOGICAL TM_NUMBER                ! ** TMAP mod 11/94 **

	NUM=.FALSE.
C
C	CHECK FOR STRING
C
	IER=0
	SLEN=LNBLK(STRING,SLEN)
	STRING=STRING(:SLEN)
	IF(STRING(1:1).EQ.'"')THEN
	    IF(STRING(SLEN:SLEN).NE.'"')THEN
		IER=7
		RETURN
	    ENDIF
	    TEMP=STRING(2:SLEN-1)
	    SLEN=SLEN-2
	ELSE
C
C	CHECK FOR NUMBER
C
	    if(slen.eq.1)then

#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(string) ) GOTO 100 ! TMAP Mod
#endif
		read(string,'(i1)',err=100)jk
		x=jk
	    else
	        WRITE(FRMT,999)SLEN
999	    FORMAT('(E',I3.3,'.0)')

#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(string) ) GOTO 100 ! TMAP Mod
#endif
	    READ(STRING,FRMT,ERR=100)X
	    endif
	    NUM=.TRUE.
	    RETURN
	ENDIF
200	IF(SLEN.EQ.0)GOTO 300
	if(slen.eq.1)then
#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(temp) ) GOTO 300 ! TMAP Mod
#endif
	    read(temp,'(i1)',err=300)jk
	    x=jk
	else
	    WRITE(FRMT,999)SLEN
#ifdef AIX_XLF
            IF ( .NOT.TM_NUMBER(temp) ) GOTO 300 ! TMAP Mod
#endif
	    READ(TEMP,FRMT,ERR=300)X
	endif
	NUM=.TRUE.
300	STRING=TEMP
	RETURN
100	CALL GETSYM(STRING(:30),TEMP,SLEN,IER2)
	IF(IER2.NE.0)THEN
	    STRING=' '
	    SLEN=0
	    RETURN
	ENDIF
	GOTO 200
	END
