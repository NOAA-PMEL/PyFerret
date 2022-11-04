	SUBROUTINE RDSEQ(X,NSIZE,PBUF,IBASE,FILE)
	DIMENSION X(NSIZE)
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
C**    @(#)rdseq.F	1.2    3/18/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V695  *acm*  9/15 Changes for hour:min labeling on time axes

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'HD.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'AXISL.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'CMRDL.INC'
	INCLUDE 'DATA.INC'
	INCLUDE 'EPICLUN.INC'
	INCLUDE 'LINES.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'LUNITS.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'SWITCH.INC'
	INCLUDE 'COMTHDR.INC'
	INCLUDE 'COMEPV.INC'
	INCLUDE 'COMEPS.INC'

	REAL*8	DIFF
	LOGICAL LEOF,LFIRST,LSTICK
	CHARACTER*10 CEPIC1,CEPIC2,FILE*(*)
	CHARACTER YES*3,NO*2,SYM*120,CSYM*6
	INTEGER PBUF
	DATA YES,NO/'YES','NO'/
C
	print *, 'SUBROUTINE RDSEQ'
	SYM='*PPL$EOF'
	CALL PUTSYM(SYM,NO,2,IER)
C
	IF(LINEN.EQ.0)CALL RSMNMX
	I=1
	IF(IFLG(4).NE.0.AND.M.GE.4)I=P(4)
	ITYPEZ=0
	IF(IFLG(3).NE.0.AND.M.GE.3)ITYPEZ=IFIX(P(3))
	ILOOP=I
	IF (IEFLG.EQ.1 .AND. IEPV2.EQ.10) THEN
	    ILOOP=I*2
	    LSTICK=.TRUE.
	ELSE
	    LSTICK=.FALSE.
	ENDIF
	LFIRST=.TRUE.
C_______
	DO 1603 JJ=1,ILOOP
	  IF(.NOT.SVECTR)THEN
	    IF(IFLG(1).EQ.0)THEN
		NX=NSIZE/2-IBASE
	    ELSE
		NX=IFIX(P(1))
	    ENDIF
	    IF(IFLG(2).EQ.0.OR.M.LT.2)THEN
		NY=1
	    ELSE
		NY=IFIX(P(2))
	    ENDIF
	    IF(JZ.NE.0)THEN
		IF(NX*NY.GT.NSIZE)THEN
		    IF(.NOT.QUIETF)WRITE(LTTOUT,9993)NX*NY
9993		    FORMAT(I10,' exceeds buffer')
		    GOTO 10
		ENDIF
	    ELSE
		NX=MIN0(NSIZE/2-IBASE,NX)
	    ENDIF
	  ENDIF
	    NREC=NX*NY/NGRP
	    LINEN=LINEN+1
	    IF(IFORM(1:3).EQ.'DSF'.OR.IFORM(1:4).EQ.'BIBO'
     *		.OR.IFORM(1:3).EQ.'PPL')THEN
		CALL REDSF(X,NX,NY,NSIZE,IBASE,FILE)
C
	    ELSE IF(IFORM(1:3).EQ.'EPI')THEN
		IF (LSCTD) THEN
		    IF (LSNEXT) BACKSPACE (LUN11)
		    CALL EPICRD (NCAST,NDATA,LEOF,.FALSE.)
		    DO 1050 IDUM1=1,8
1050		    THEADER(IDUM1)=HEADER(IDUM1)
		ELSE
		    IF (LSNEXT) BACKSPACE (LUN11)
		    CALL EPICTRD(IDUM1,XDATE,XTIME,XDAT2,XTIM2,
     1		            P(14),LEOF,.FALSE.)
		    IF (P(14) .LE. 0.) THEN
			P(14)=-P(14)
			LSTIME=.TRUE.
		     ENDIF
		ENDIF
		IF(LEOF)THEN
		    LINEN=LINEN+1
		    LLENG(LINEN)=0
		    GOTO 110
		ENDIF
C
C  GET U FOR STICK PLOT, SET UP FOR V ON NEXT PASS
C
		IF (LSTICK .AND. LFIRST) THEN
		    IEPV1=9
		    IEPV2=6
		    BACKSPACE (LUN11)
C
C  GET V FOR STICK PLOT
C
		ELSE IF (LSTICK) THEN
		    IEPV1=9
		    IEPV2=7
		ENDIF
		IF (IEFLG .EQ. 1) CALL EPICVC
C
C           JX=JY=0 IMPLIES BOTH AXES ARE TIME AXES.  BYPASS THIS
C           DATA SET.
C
		IF (JY.EQ.0 .AND. JX.EQ.0) THEN
		    LINEN=LINEN+1
		    LLENG(LINEN)=0
		    GOTO 110
		ENDIF
		IF (IGTLAB .EQ. 1) THEN
		    CALL EPICTLAB
		    IF (LSTICK .AND. (.NOT.LFIRST)) THEN
			IYLAB=MAX0 (INDEX(YLAB,',')+1, INDEX(YLAB,'(')-1)
			YLAB(1:)=YLAB(IYLAB:)
		    ENDIF
		ENDIF
		IF(ITFLG .NE. 0)THEN
C
C		Get end time adjusted by time offset if needed.
C
		    IF (LSOFFS) THEN
			CALL EPICTIMC (XDAT2,XTIM2,CEPIC1)
			CALL TIME1 (1,CEPIC1, 2,'          ',
     1			            ICEPIC2,CEPIC2,P(14)/2.)
			READ (CEPIC2,2003) XDAT2,XTIM2
2003			FORMAT (F6.0,F4.0)
		    ENDIF
C
C		Get time axis limits and set up for time variable.
C
		    CALL EPICTLIM (XDATE,XTIME,XDAT2,XTIM2)
		    CALL EPICTIMC (XDATE,XTIME,ICOM)
		    OFF=DIFF(ITSTRT,ICOM)/DT+1.
		    P(14)=P(14)/DT
		    IF (LSOFFS) OFF=OFF+P(14)/2.
		ENDIF
		IX=IBASE-1
		CALL READSQ(X,NX,NY,NSIZE,IBASE,FILE)
		IF (LSTICK .AND. LFIRST) REWIND LUN1
C
C	    SET NX TO ACTUAL NUMBER OF POINTS READ
C
		NX=LLENG(LINEN)
C
C	    IF SWITCH STIME IS FALSE, THE SEQUENCE NUMBER IS
C	    TRANSFORMED INTO UNITS OF TSTRT AND DT.
C
C	    IF SWITCH STIME IS TRUE (MEANING TO READ DATE/TIME WORDS
C	    FROM THE EPIC DATA FILE) THEN THE DATE/TIME WORDS WERE
C	    TRANSFORMED IN ROUTINE READSQ WHERE THE DATA FILE WAS READ.
C
		IF (ITFLG.EQ.1 .AND. .NOT.LSTIME) THEN
		    DO 2001 J=1,NX
			X(IX+J)=(FLOAT(J)-1.0)*P(14)+OFF
2001		    CONTINUE
		ELSE IF (ITFLG.EQ.-1 .AND. .NOT.LSTIME) THEN
		    IY=NSIZE/2 + IX
		    DO 2002 J=1,NX
			X(IY+J)=(FLOAT(J)-1.0)*P(14)+OFF
2002		    CONTINUE
		ENDIF
		IF (LSTICK) THEN
		    IEPV2=10
		    IEPV1=9
		    LFIRST=.NOT. LFIRST
		ENDIF
C
	    ELSE
		CALL READSQ(X,NX,NY,NSIZE,IBASE,FILE)
	    ENDIF
110	    IF(LLENG(LINEN).LE.0)THEN
		LINEN=LINEN-1
		IF(.NOT.QUIETF)WRITE(LTTOUT,'('' NO POINTS READ'')')
		IF(LINEN.EQ.0)PBUF=0
		SYM='*PPL$EOF'
		CALL PUTSYM(SYM,YES,3,IER)
		GOTO 10
	    END IF
	    IF(JTYPE.EQ.1)GOTO 1602
	    IF(.NOT.QUIETF)WRITE(LTTOUT,'('' CONTOUR DATA READ'')')
	    IF(.NOT.QUIETF)WRITE(LTTOUT,'('' NX='',I5,''  NY='',I5)')NX,NY
	    LNUM=1
	    IBASE=1
	    LINEN=0
	    CALL STMNMX(X,NSIZE,XMIN,XMAX,YMIN,YMAX)
	    GOTO 10
1602	    IF(.NOT.QUIETF)WRITE(LTTOUT,9990)LLENG(LINEN)
9990	    FORMAT(1X,I6,' POINTS READ')
	    WRITE (CSYM,9992)LLENG(LINEN)
9992	    FORMAT (I6)
	    SYM='*PPL$POINTS'
	    CALL PUTSYM(SYM,CSYM,6,IER)
	    PBUF=1
	    CALL STMNMX(X,NSIZE,XMIN,XMAX,YMIN,YMAX)
C
C	IF THE NEXT LINE WILL OVERFLOW THE BUFFER
C	SET UP LINEN SUCH THAT IT OVER WRITES THE LAST
C	LINE
C
	    LNUM=LINEN
	    IF(LINEN.GE.NLINES)THEN
		LINEN=NLINES - 1
	    ELSE
		IBASE=IBASE+LLENG(LINEN)
	    ENDIF
	    IF(.NOT.QUIETF)WRITE(LTTOUT,9991)LINEN
9991	    FORMAT(' LINE',I3,' READ')
1603	CONTINUE
C_______
10	RETURN
	
	ENTRY RDSEQ2(X,NSIZE,PBUF,IBASE,FILE)
#ifdef unix
	IF(.NOT.QUIETF)WRITE(LTTOUT,9990)LLENG(LINEN)
	PBUF=1
	CALL STMNMX(X,NSIZE,XMIN,XMAX,YMIN,YMAX)
C
C	IF THE NEXT LINE WILL OVERFLOW THE BUFFER
C	SET UP LINEN SUCH THAT IT OVER WRITES THE LAST
C	LINE
C
	LNUM=LINEN
	IF(LINEN.GE.NLINES)THEN
	    LINEN=NLINES - 1
	ELSE
	    IBASE=IBASE+LLENG(LINEN)
	ENDIF
	IF(.NOT.QUIETF)WRITE(LTTOUT,9991)LINEN
	GOTO 10
#else
	GOTO 1602
#endif
	END
