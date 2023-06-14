	SUBROUTINE PARSEV(LINE,UPLINE,ILEN,RESULT,IER,IPOS)
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
C**		line -- input line (char*2048)
C**		upline -- input line capitals (char*2048)
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
C* v552 *acm* 3/03 Longer strings: changes for multi-line labels
* v580 *acm* 8/04 Change use of 256 to buflen+1, 
*                 where buflen is defined in cmndlen.inc
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* 12/2022 *acm* Code cleanup: Rewrite to replace the computed GOTO statement 
*                             "GOTO(201,202,203,...)ISW"  with IF's

        INCLUDE 'cmndlen.inc'

	CHARACTER LINE*(*),UPLINE*(*),EXP1*2048,EXP2*2048
	LOGICAL RESULT,NUM1,NUM2,NUM
	IER=0
C
C	CHECK FOR OPERATORS
C
	IEQ=INDEX(UPLINE,'.EQ.')
	IF(IEQ.LE.0)THEN
	    IEQ=buflen+1
	ELSE
	    ISW=1
	ENDIF
	INE=INDEX(UPLINE,'.NE.')
	IF(INE.LE.0)THEN
	    INE=buflen+1
	ELSE
	    ISW=2
	ENDIF
	ILT=INDEX(UPLINE,'.LT.')
	IF(ILT.LE.0)THEN
	    ILT=buflen+1
	ELSE
	    ISW=3
	ENDIF
	IGT=INDEX(UPLINE,'.GT.')
	IF(IGT.LE.0)THEN
	    IGT=buflen+1
	ELSE
	    ISW=4
	ENDIF
	ILE=INDEX(UPLINE,'.LE.')
	IF(ILE.LE.0)THEN
	    ILE=buflen+1
	ELSE
	    ISW=5
	ENDIF
	IGE=INDEX(UPLINE,'.GE.')
	IF(IGE.LE.0)THEN
	    IGE=buflen+1
	ELSE
	    ISW=6
	ENDIF
	IOP=MIN0(IEQ,INE,ILT,IGT,ILE,IGE)
	IF(IOP.GE.buflen+1)THEN
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

	   IF (ISW .EQ. 1) RESULT=X1.EQ.X2
	   IF (ISW .EQ. 2) RESULT=X1.NE.X2
	   IF (ISW .EQ. 3) RESULT=X1.LT.X2
	   IF (ISW .EQ. 4) RESULT=X1.GT.X2
	   IF (ISW .EQ. 5) RESULT=X1.LE.X2
	   IF (ISW .EQ. 6) RESULT=X1.GE.X2

	ELSE

	   IF (ISW .EQ. 1) RESULT=EXP1(:LEN1).EQ.EXP2(:LEN2)
	   IF (ISW .EQ. 2) RESULT=EXP1(:LEN1).NE.EXP2(:LEN2)
	   IF (ISW .EQ. 3) RESULT=EXP1(:LEN1).LT.EXP2(:LEN2)
	   IF (ISW .EQ. 4) RESULT=EXP1(:LEN1).GT.EXP2(:LEN2)
	   IF (ISW .EQ. 5) RESULT=EXP1(:LEN1).LE.EXP2(:LEN2)
	   IF (ISW .EQ. 6) RESULT=EXP1(:LEN1).GE.EXP2(:LEN2)

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
C**		string -- input expression (char*2048)
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
*       *jd* 11.94 - modified for AIX port
*       Modification uses TM_NUMBER to determine if "value" is a number rather
*       than the ERR= branch of a READ which fails on AIX/xlf

C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER STRING*(*),TEMP*2048,FRMT*120
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
		read(string,'(i1)',err=100)jk
		x=jk
	    else
	        WRITE(FRMT,999)SLEN
999	    FORMAT('(E',I3.3,'.0)')
	    READ(STRING,FRMT,ERR=100)X
	    endif
	    NUM=.TRUE.
	    RETURN
	ENDIF
200	IF(SLEN.EQ.0)GOTO 300
	if(slen.eq.1)then
	    read(temp,'(i1)',err=300)jk
	    x=jk
	else
	    WRITE(FRMT,999)SLEN
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
