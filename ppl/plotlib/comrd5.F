      SUBROUTINE COMRD5(ICOM,X,M,ILB,N,IFLAG) 
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
C**    @(#)comrd5.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*     *jd 11.94 - modified for AIX port
*       Modification uses TM_NUMBER to determine if "value" is a number rather
*       than the ERR= branch of a READ which fails on AIX/xlf
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

      CHARACTER ICOM*(*),ILB*(*),IN*80,LT*1,JX*20,IF*8,IA*1 
      CHARACTER NAME*81,INTEMP*80,PRMPT*40
      DIMENSION X(*),IFLAG(*)
#ifndef unix
	byte iab
	equivalence (iab,ia)
#endif 
      COMMON/CEFTN5/ IN,IA,LT,NAME,INTEMP
      COMMON/CDFTN5/LN 
      COMMON/CPFTN5/PRMPT


#ifdef unix
CC   If unix, initialize IN in block data stmt in cmrdst.F.  The block data
CC   is declared external in subroutine CMRDST in that file.
      DATA J,KM/0,80/
#else
      DATA J,KM,IN/0,80,' '/ 
#endif

C 
      LOGICAL TM_NUMBER                ! ** TMAP mod 11/94 **
C
C     THIS ROUTINE WAS WRITTEN BY DON DENBO IN MAY, 1980 
C     IT IS LIKE COMRD BUT WRITTEN IN FORTRAN 77 
C     SEE MILELIB DOC FOR DETAILS. 
C     MODIFIED FOR PDP 11 IN AUGUST 1983 BY D. DENBO
C 
      ICOM=' ' 
      ILB=' ' 
      M=0 
      N=0 
      IFLAG(1)=0 
      IF(J.NE.0)GOTO 21 
#ifdef unix
	IF(NAME(1:3).EQ.'/dev/tty')WRITE(LN,PRMPT)
#else
	IF(NAME(1:3).EQ.'TT:')WRITE(LN,PRMPT)
#endif
      READ(LN,'(A80)',END=901)IN 
      DO 20 KM=80,1,-1 
      IF(IN(KM:KM).NE.' ')GOTO 21 
20    CONTINUE 
21	K=INDEX(IN(J+1:),LT)+J
	IF(K.EQ.J)K=81
	INTEMP=IN(J+1:K-1)
 	CALL NEFTN5(J,KM,IS)
	GOTO (200,31,22,23,900),IS
22    M=1 
      GOTO 31 
23    J0=J 
#ifndef unix
	jb=iab
	iab=iand(jb,95)
#endif
      ICOM=IA 
      DO 30 I=2,10 
	CALL NEFTN5(J,KM,IS)
#ifdef unix
	GOTO (30,33,33,30,33),IS
#else
	goto (29,33,33,29,33),is
29	jb=iab
	iab=iand(j,95)
#endif
30	ICOM(I:I)=IA
      J=J0 
      ICOM=' ' 
      GOTO 32 
33    IF(IA.EQ.LT.OR.J.GT.KM)GOTO 900 
31	CALL NEFTN5(J,KM,IS)
	GOTO (200,31,32,300,900),IS
32    M=M+1 
      IFLAG(M)=0 
      GOTO 31 
300   ILB=IA 
      N=1 
      DO 60 J=J+1,KM 
      IA=IN(J:J) 
      IF(IA.EQ.LT)GOTO 900 
      N=N+1 
	ILB(N:N)=IA
60    CONTINUE 
      GOTO 900 
200   M=M+1 
      JX=IA 
      K=1 
90	CALL NEFTN5(J,KM,IS)
	GOTO (80,81,81,80,81),IS
 80    K=K+1 
	JX(K:K)=IA
      GOTO 90 
81	WRITE(IF,'(''(E'',I2,''.0)'')')K
      IFLAG(M)=1 

#ifdef AIX_XLF
      IF ( .NOT.TM_NUMBER(JX) ) GOTO 40  ! ** TMAP mod 11/94 **
#endif
      READ(JX,IF,ERR=40)X(M) 
40    IF(IA.EQ.LT.OR.J.GT.KM)GOTO 900 
      IF(IA.EQ.',')GOTO 31 
100	CALL NEFTN5(J,KM,IS)
	GOTO (200,100,31,300,900),IS
900   IF(IA.EQ.LT)RETURN 
      J=0 
      RETURN 
c
c	eof read
c
901   continue
c	CLOSE(LN) 
c	OPEN(LN,FILE=NAME,STATUS='UNKNOWN') 
      RETURN 
      END 
