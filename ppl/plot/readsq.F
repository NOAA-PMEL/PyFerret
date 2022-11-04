        SUBROUTINE READSQ(X,NXS,NYS,NSIZE,IBASE,NAME)
	DIMENSION X(NSIZE),XT(1000,20)
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
C**    @(#)readsq.F	1.2    3/18/88
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
#ifdef unix
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'zgridd_inc.decl'
        include 'ZGRIDD.INC'
        include 'zgrid_inc.decl'
        include 'ZGRID.INC'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'hd_inc.decl'
        include 'HD.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'cmrd_inc.decl'
        include 'CMRD.INC'
        include 'cmrdl_inc.decl'
        include 'CMRDL.INC'
        include 'labels_inc.decl'
        include 'LABELS.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
        include 'lunits_inc.decl'
        include 'LUNITS.INC'
        include 'system_inc.decl'
        include 'SYSTEM.INC'
        include 'comeps_inc.decl'
        include 'COMEPS.INC'
        include 'comthdr_inc.decl'
        include 'COMTHDR.INC'
        include 'taxis_inc.decl'
        include 'TAXIS.INC'
        include 'axisl_inc.decl'
        include 'AXISL.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
#else
        INCLUDE 'tmap_pplv11inc:tmap_PARAMPL5.DAT'
        INCLUDE 'tmap_pplv11inc:tmap_ZGRIDD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_ZGRID.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LINES.INC'
        INCLUDE 'tmap_pplv11inc:tmap_HD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_DATA.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRDL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LABELS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_MISS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LUNITS.INC'
        INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
        INCLUDE 'PPLV11INC:SYSTEM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_COMEPS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_COMTHDR.INC'
        INCLUDE 'tmap_pplv11inc:tmap_TAXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_AXISL.INC'
        INCLUDE 'TMAP_PPLV11INC:SWITCH_INC.DECL'
        INCLUDE 'PPLV11INC:SWITCH.INC'
#endif
	INTEGER*4 IEPIC1,IEPIC2
	CHARACTER NAME*(*)
C     NREC=NUMBER OF RECORDS TO BE READ
C     NGRP=NUMBER OF GROUPS TO BE READ PER RECORD
C     NVAR=NUMBER OF VARIABLE PER GROUP


	print *, 'SUBROUTINE READSQ'
	IX=IBASE-1
	IY=NSIZE/2+IX
	IF(SVECTR)THEN
	    IVLEN=NX*NY
	    IF(JX.GT.0.and.JZ.GT.0)THEN
		IF(IZLEN.EQ.0)IZLEN=IZSE
	    ELSE
		IZLEN=NX*NY
	    ENDIF
	    NREC=IZLEN/NGRP
	ELSE
	    IVLEN=0
	    IZLEN=0
	    IF(JX.GT.0.AND.JZ.GT.0)THEN
	    	CALL STPRMP('rd>',3)
	    	IF(.NOT.QUIETF)WRITE(LTTOUT,'('' ENTER # PTS TO READ'')')
	     	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	    	CALL STPRMP('ppl+>',5)
C
C	RSX-11M
C
C	CLOSE(6)
C	OPEN(LMULT,FILE='SY:QZZDAT.DAT',STATUS='NEW',FORM='UNFORMATTED')
C
C	END RSX-11M
C
	    	IF(IFLG(1).EQ.0)P(1)=IZSE
	    	NREC=MIN0(IZSE,INT(P(1)))/NGRP
	    END IF
	ENDIF
	II=1
	JJ=1
	DO 50 KK=1,NREC
	IF(IFORM(1:3).NE.'UNF'.AND.IFORM(1:3).NE.'EPI')THEN
	    IF(IFORM(1:4).EQ.'FREE')THEN
		READ(LUNN,*,END=11,ERR=1000)((XT(K,J),J=1,NVAR),
     *		K=1,NGRP)
	    ELSE
		READ(LUNN,IFORM,END=11,ERR=1000)((XT(K,J),J=1,NVAR),
     *		K=1,NGRP)
	    END IF
	ELSE
C     IF UNFORMATTED USE NGRP OR NREC WHICH EVER IS LARGER
	    READ(LUNN,END=11,ERR=1000)((XT(K,J),J=1,NVAR),K=1,NGRP)
	END IF
	IF(JZ.GT.0)GOTO 100
	IF(JX.GT.0)THEN
	    IF(JY.GT.0)THEN
		DO 20 I=1,NGRP
		X(IX+II)=XT(I,JX)
		X(IY+II)=XT(I,JY)
20		II=II+1
	    ELSE
C
C		If EPIC data and switch STIME (read EPIC time words),
C		make time variable = a sequence number which is 1 for
C		ITSTRT (usually first data point) and for which an
C		increment of 1 is 1 DT unit (usually 1440 minutes).
C		Offset by epico, which is delta-t for time series
C		on data file header (to get to center of time interval
C		if that is non-zero), unless lsoffs=.false.
C
		IF (LSTIME .AND. IFORM(1:3).EQ.'EPI') THEN
		    IF (LSOFFS) THEN
		        READ (THEADER(3)(61:78),222) EPICO
222		        FORMAT (E18.0)
		        EPICO=EPICO/2./DT
		    ELSE
		        EPICO=0.
		    ENDIF
		    READ (ITSTRT,221) EPDATE,EPTIME
221		    FORMAT (F6.0,F4.0)
		    CALL EPICSQ (EPDATE,EPTIME,IEPIC1)
		    DO 200 I=1,NGRP
		    X(IX+II)=XT(I,JX)
		    CALL EPICSQ (XT(I,1),XT(I,2),IEPIC2)
		    X(IY+II) = FLOAT(IEPIC2-IEPIC1) / DT + 1. + EPICO
200		    II=II+1
		ELSE
		    DO 21 I=1,NGRP
		    X(IX+II)=XT(I,JX)
		    X(IY+II)=FLOAT(II)
21		    II=II+1
	    	ENDIF
	    END IF
	ELSE
	    IF (LSTIME .AND. IFORM(1:3).EQ.'EPI') THEN
		IF (LSOFFS) THEN
		    READ (THEADER(3)(61:78),222) EPICO
		    EPICO=EPICO/2./DT
		ELSE
		    EPICO=0.
		ENDIF
		READ (ITSTRT,221) EPDATE,EPTIME
		CALL EPICSQ (EPDATE,EPTIME,IEPIC1)
	        DO 210 I=1,NGRP
		CALL EPICSQ (XT(I,1),XT(I,2),IEPIC2)
		X(IX+II) = FLOAT(IEPIC2-IEPIC1) / DT + 1. + EPICO
	        X(IY+II)=XT(I,JY)
210	        II=II+1
	    ELSE
	        DO 22 I=1,NGRP
	        X(IX+II)=FLOAT(II)
	        X(IY+II)=XT(I,JY)
22	        II=II+1
	    ENDIF
	END IF
	GOTO 50
100	IF(JX.LE.0)THEN
	    DO 30 IC=1,NGRP
	    X(IVLEN+NXS*(JJ-1)+II)=XT(IC,JZ)
	    IF(ITYPEZ.EQ.0)THEN
		II=II+1
		IF(II.GT.NX)THEN
		    JJ=JJ+1
		    II=1
		ENDIF
	    ELSE
		JJ=JJ+1
		IF(JJ.GT.NY)THEN
		    II=II+1
		    JJ=1
		ENDIF
	    END IF
30     CONTINUE
	ELSE
	    DO 31 IC=1,NGRP
C
C	RSX-11M
C
C	WRITE(LMULT)XT(IC,JX),XT(IC,JY),XT(IC,JZ)
C
C	END RSX-11M
C
C	VAX/VMS
C
	    XX(II)=XT(IC,JX)
	    YY(II)=XT(IC,JY)
	    ZZ(II)=XT(IC,JZ)
	    II=II+1
C
C	END VAX/VMS
C
31	    CONTINUE
	ENDIF
50	CONTINUE
11	LEN=(KK-1)*NGRP
	LLENG(LINEN)=LEN
	IF(LEN.LE.0)RETURN
	IF(JZ.GT.0)GOTO 400
	JTYPE=1
	ZMIN=0.
	ZMAX=ZMIN
	XMIN=1.E38
	YMIN=XMIN
	XMAX=-XMIN
	YMAX=XMAX
	DO 23 I=1,LEN
	IF((XLE.AND.X(IX+I).LE.CMXLE).OR.
     *	   (XEQ.AND.X(IX+I).EQ.CMXEQ).OR.
     *	   (XGE.AND.X(IX+I).GE.CMXGE).OR.
     *	   (YLE.AND.X(IY+I).LE.CMYLE).OR.
     *	   (YEQ.AND.X(IY+I).EQ.CMYEQ).OR.
     *	   (YGE.AND.X(IY+I).GE.CMYGE))GOTO 23
	IF(X(IX+I).LT.XMIN)XMIN=X(IX+I)
	IF(X(IX+I).GT.XMAX)XMAX=X(IX+I)
	IF(X(IY+I).LT.YMIN)YMIN=X(IY+I)
	IF(X(IY+I).GT.YMAX)YMAX=X(IY+I)
23	CONTINUE
	RETURN
400	JTYPE=0
	IF(SVECTR)JTYPE=-1
C
C	RSX-11M
C
C	IF(JX.GT.0)THEN
C	CLOSE(6)
C	OPEN(LMULT,FILE='SY:ECHO.DAT',STATUS='OLD',ACCESS='APPEND',
C     *	CARRIAGECONTROL='LIST')
C	ENDIF
C
C	END RSX-11M
C
	IF(.NOT.SVECTR)THEN
	    CALL STPRMP('rd>',3)
	    IF(.NOT.QUIETF)WRITE(LTTOUT,'('' ENTER XMIN,XMAX,YMIN,YMAX'')')
	    CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	    CALL STPRMP('ppl+>',5)
	    XMIN=P(1)
	    XMAX=P(2)
	    YMIN=P(3)
	    YMAX=P(4)
	ENDIF
	IF(JX.GT.0)THEN
	    DX=(XMAX-XMIN)/FLOAT(NX-1)
	    DY=(YMAX-YMIN)/FLOAT(NY-1)
	    IZLEN=LEN
C
C	RSX-11M
C
C	CLOSE(LMULT)
C	OPEN(LMULT,FILE='SY:QZZCMD.DAT',STATUS='NEW',FORM='UNFORMATTED')
C	WRITE(LMULT)LEN,NX,NY,XMIN,YMIN,DX,DY,CAY,NRNG
C	CLOSE(LMULT)
C	CALL GETLUN(LTTOUT,DAT)
C	WRITE(DV,'(I2)')DEV
C	IF(DEV.LT.10)DV(1:1)='0'
C	CMDLIN(28:29)=DV
C	CALL SPAWN(MCR,,,,,,,CMDLIN,29)
C	CALL STOP
C	OPEN(LMULT,FILE='SY:QZZDAT.DAT',STATUS='OLD',FORM='UNFORMATTED')
C	DO 500 I=1,NX
C	DO 500 J=1,NY
C	READ(LMULT)X(NXS*(J-1)+I)
C500	CONTINUE
C	CLOSE(LMULT,DISP='DELETE')
C	OPEN(LMULT,FILE='SY:ECHO.DAT',STATUS='OLD',ACCESS='APPEND',
C     *	CARRIAGECONTROL='LIST')
C
C	END RSX-11M
C
C	VAX/VMS
C
	    DO 500 I=1,NX
	    DO 500 J=1,NY
	    X(IVLEN+NXS*(J-1)+I)=0.0
500	    CONTINUE
	    IF(CAY.GE.0)THEN
		CALL ZGRID(X(IVLEN+1),NX,NY,NX,NY,XMIN,YMIN,DX,DY,
     *		XX,YY,ZZ,LEN,CAY,NRNG)
	    ELSE
		CALL LINEAR(X(IVLEN+1),NX,NY,NX,NY,XMIN,YMIN,DX,DY,
     *		XX,YY,ZZ,LEN,CAY,NRNG)
	    ENDIF
C
C	END VAX/VMS
C
	ENDIF
	RETURN
C	ERRORS
1000	IF(.NOT.QUIETF)WRITE(LTTOUT,
     *	'('' FATAL READ ERRORS IN READSEQ'')')
	IF(.NOT.QUIETF)WRITE(LTTOUT,9999) NAME,IFORM
9999	FORMAT('  NAME= ',A30,'   TYPE= ',A20)
#ifndef unix
	CALL ERRTST(64,I)
#endif
	RETURN
	END
