	SUBROUTINE SETSYM(LINE,ILEN,IER,IST)
C** 
C**    @(#)setsym.f	1.2    5/26/88
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
C**		CALL SETSYM(LINE,ILEN,IER,IST)
C**
C**	Parameters:
C**		LINE -- Line containing symbol and value (char*2048)
C**		ILEN -- Length of LINE
C**		IER  -- Error code
C**			0 = success
C**			1 = symbol not found
C**		IST -- relative position of sym in LINE
C**
C**	Description:
C**		Parses LINE and sets the symbol defined with the value
C**		specified.
C**
C**	History:
C**	    Written:
C**		22-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
** TMAP mod **
* *jd* 11.3.94 Fixed(?) the fix -- permit read if text is a number
**************
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

C* v552 *acm* 3/03 Longer strings: changes for multi-line labels
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* v695 9/15 *acm* write values to PPL* symbols using 6 digits not 4
* v721 8/17 *kms* always do TM_NUMBER check for newer gfortran (Mac)


	CHARACTER LINE*(*),VALUE*2048,SYM*120,OPR*1,ARG(4)*120
	CHARACTER FUNCT*30,FRMT*30
	LOGICAL NUM,FUNC

	LOGICAL TM_NUMBER		! ** TMAP mod 10/94 **
C
C	PARSE LINE 
C
	IST=0
	NUM=.TRUE.
	ISPC=INDEX(LINE(:ILEN),' ')
	IF(ISPC.EQ.0.)ISPC=ILEN+1
	IPARN=INDEX(LINE(:ILEN),')')
	IF(IPARN.EQ.0.OR.IPARN.GT.ISPC)THEN
	    I=ISPC-1
	ELSE
	    I=IPARN
	ENDIF
	SYM=LINE(:I)
	N=0
100	I=I+1
	IF(I.GT.ILEN)GOTO 120
	IF(LINE(I:I).EQ.' ')GOTO 100
C
C	I=START OF VALUE
C
	FUNC=LINE(I:I).EQ.'$'
C
	IF(LINE(I:I).EQ.'"')THEN
	    I=I+1
	    NUM=.FALSE.
	ENDIF
	VALUE=LINE(I:ILEN)
	N=ILEN-I+1
	IF(VALUE(N:N).EQ.'"')THEN
	    VALUE(N:N)=' '
	    N=N-1
	ENDIF
C
C	TEST FOR ALGEBRA AND DECODE IF IT HAS CORRECT SYNTAX
C
	IF(FUNC)THEN
	    FUNCT=VALUE
	    CALL UPPER(FUNCT,30)
C
C	PARSE FUNCTION ARGUEMENTS
C
	    I=INDEX(VALUE,'(')+1
	    J=INDEX(VALUE,')')-1
	    VALUE=VALUE(I:J)
	    N=J-I+1
	    K=1
300	    CALL CMLJST(VALUE,N)
	    IF(VALUE(1:1).EQ.'"')THEN
		II=INDEX(VALUE(2:),'"')
		ARG(K)=VALUE(2:II)
		I=INDEX(VALUE(II+1:),',')+II-1
		IF(I.EQ.II-1)I=N
	    ELSE
		I=INDEX(VALUE,',')-1
		IF(I.EQ.-1)I=N
		ARG(K)=VALUE(:I)
	    ENDIF
	    IF(I.LT.N)THEN
		K=K+1
		VALUE=VALUE(I+2:)
		N=N-I-1
		GOTO 300
	    ENDIF
C
C	$EDIT(symbol,arguement)
C
	    IF(FUNCT(1:3).EQ.'$ED')THEN
		CALL GETSYM(ARG(1),VALUE,N,IER)
		CALL UPPER(ARG(2),30)
		IF(INDEX(ARG(2),'UP').NE.0)THEN
		    CALL UPPER(VALUE,N)
		ENDIF
		IF(INDEX(ARG(2),'TR').NE.0)THEN
		    CALL CMLJST(VALUE,N)
		    N=LNBLK(VALUE,N)
		ENDIF
		IF(INDEX(ARG(2),'COM').NE.0)THEN
		    CALL CMPRSS(VALUE,N)
		ENDIF
		IF(INDEX(ARG(2),'COL').NE.0)THEN
		    CALL SQUISH(VALUE,1,N)
		    N=LNBLK(VALUE,N)
		ENDIF
C
C	$EXTRACT(start,length,symbol)
C
	    ELSE IF(FUNCT(1:3).EQ.'$EX')THEN
		CALL GETSYM(ARG(3),VALUE,N,IER)
		CALL RDFLT(ARG(1),X)
		I=INT(X)
		CALL RDFLT(ARG(2),X)
		J=INT(X)
		VALUE=VALUE(I:I+J)
		N=J
C
C	$INTEGER(symbol)
C
	    ELSE IF(FUNCT(1:3).EQ.'$IN')THEN
		CALL GETSYM(ARG(1),VALUE,N,IER)
		CALL RDFLT(VALUE,X)
		WRITE(VALUE,'(I10)')INT(X+0.5)
		N=10
		CALL CMLJST(VALUE,N)
		N=LNBLK(VALUE,N)
C
C	$LENGTH(symbol)
C
	    ELSE IF(FUNCT(1:3).EQ.'$LE')THEN
		CALL GETSYM(ARG(1),VALUE,N,IER)
		WRITE(VALUE,'(I10)')N
		N=10
		CALL CMLJST(VALUE,N)
		N=LNBLK(VALUE,N)
C
C	$LOCATE(substring,symbol)
C
	    ELSE IF(FUNCT(1:3).EQ.'$LO')THEN
		CALL GETSYM(ARG(2),VALUE,N,IER)
		K=LNBLK(ARG(1),30)
		I=INDEX(VALUE(:N),ARG(1)(:K))
		WRITE(VALUE,'(I10)')I
		N=10
		CALL CMLJST(VALUE,N)
		N=LNBLK(VALUE,N)
C
C	$ELEMENT(position,delimeter,symbol)
C
	    ELSE IF(FUNCT(1:3).EQ.'$EL')THEN
		CALL GETSYM(ARG(3),VALUE,N,IER)
		CALL RDFLT(ARG(1),X)
		K=INT(X)
		I=1
400		CALL CMLJST(VALUE,N)
		J=INDEX(VALUE(:N),ARG(2)(1:1))
		IF(J.EQ.0)THEN
		    IF(K.EQ.I)THEN
			J=N+1
		    ELSE
			VALUE=ARG(2)(1:1)
			N=1
			GOTO 410
		    ENDIF
		ENDIF
		IF(K.EQ.I)THEN
		    VALUE=VALUE(:J-1)
		    N=LNBLK(VALUE,J-1)
		    GOTO 410
		ELSE
		    VALUE=VALUE(J+1:)
		    N=N-J
		    I=I+1
		ENDIF
		GOTO 400
410		CONTINUE
	    ENDIF
	ELSE IF(NUM)THEN
	    I=INDEX(VALUE,' ')
	    K=I-1
	    if(k.eq.1)then
	       IF ( .NOT.TM_NUMBER(value(:k)) ) GOTO 120! ** TMAP mod 10/94 **
		read(value(:k),'(i1)',err=120)jk
		x1=jk
	    else
	    WRITE(FRMT,999)K
999	    FORMAT('(F',I3.3,'.0)')
	    IF ( .NOT.TM_NUMBER(value(:i-1)) ) GOTO 120	! ** TMAP mod 10/94 **
	    READ(VALUE(:I-1),FRMT,ERR=120)X1
	    endif
200	    I=I+1
	    IF(I.GT.N)GOTO 120
	    IF(VALUE(I:I).EQ.' ')GOTO 200
	    OPR=VALUE(I:I)
	    IOPR=I
210	    I=I+1
	    IF(I.GT.N)GOTO 120
	    IF(VALUE(I:I).EQ.' ')GOTO 210
	    IST=INDEX(VALUE(I:),' ')+I-2
	    K=IST-I+1
	    WRITE(FRMT,999)K
	    IF ( .NOT.TM_NUMBER(value(i:ist)) ) GOTO 120	! ** TMAP mod 10/94 **
	    READ(VALUE(I:IST),FRMT,ERR=120)X2
	    IF(OPR.EQ.'+')THEN
		X=X1+X2
	    ELSE IF(OPR.EQ.'-')THEN
		X=X1-X2
	    ELSE IF(OPR.EQ.'*')THEN
		X=X1*X2
	    ELSE IF(OPR.EQ.'/')THEN
		X=X1/X2
	    ELSE
		GOTO 120
	    ENDIF
	    CALL PUTVAL(SYM,X,7,IER)
	    RETURN
	ENDIF
120	CALL PUTSYM(SYM,VALUE,N,IER)
	RETURN
	END
	SUBROUTINE CMPRSS(STR,LEN)
C
C	REMOVE EXTRA BLANKS
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STR*(*),TEMP*2048
	LOGICAL BLANK
	K=1
	BLANK=.FALSE.
	DO 100 J=1,LEN
	    IF(STR(J:J).EQ.' ')THEN
		IF(.NOT.BLANK)THEN
		    BLANK=.TRUE.
		ELSE
		    GOTO 100
		ENDIF
	    ELSE
		BLANK=.FALSE.
	    ENDIF
	    TEMP(K:K)=STR(J:J)
	    K=K+1
100	CONTINUE
	STR=TEMP(:K-1)
	LEN=K-1
	RETURN
	END
	SUBROUTINE RDINT(STR,K)
	CHARACTER STR*(*),FRMT*30
	N=LNBLK(STR,30)
	if(n.eq.0)then
	    k=0
	    return
	endif
	WRITE(FRMT,999)N
999	FORMAT('(I',I2.2,')')
	READ(STR,FRMT)K
	RETURN
	END
	SUBROUTINE RDFLT(STR,X)
        LOGICAL TM_NUMBER                ! ** TMAP mod 11/94 **
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STR*(*),FRMT*30
	N=LNBLK(STR,30)
	IF(N.EQ.0)THEN
C
C	DEFAULT VALUE IF NO NUMBER IS 1.0
C
   	    X=1.0
	    RETURN
	ENDIF
	if(n.eq.1)then
            IF ( .NOT.TM_NUMBER(str) ) GOTO 1000  ! ** TMAP mod 11/94 **
	    read(str,'(i1)',err=1000)jk
	    x=jk
	else
	WRITE(FRMT,999)N
999	FORMAT('(F',I2.2,'.0)')
            IF ( .NOT.TM_NUMBER(str) ) GOTO 1000  ! ** TMAP mod 11/94 **
	READ(STR,FRMT,ERR=1000)X
	endif

C       7/97 added following in - needed to preven reseting of x *kob*
	RETURN
C	Change above err=100 stmts to err=1000 and copy that line
C	for FORTRAN 90 on linux *jd* 1.21.97
1000	x=1.0

	RETURN
	END

