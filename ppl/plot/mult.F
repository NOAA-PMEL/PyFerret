	SUBROUTINE MULT
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
C**    @(#)mult.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
        include 'tmap_pplv11inc/axis_inc.decl'
        include 'pplv11inc/AXIS.INC'
        include 'tmap_pplv11inc/mplot_inc.decl'
        include 'pplv11inc/MPLOT.INC'
        include 'tmap_pplv11inc/cmrd_inc.decl'
        include 'pplv11inc/CMRD.INC'
        include 'tmap_pplv11inc/cmrdl_inc.decl'
        include 'pplv11inc/CMRDL.INC'
        include 'tmap_pplv11inc/plt_inc.decl'
        include 'pplv11inc/PLT.INC'
        include 'tmap_pplv11inc/filnam_inc.decl'
        include 'pplv11inc/FILNAM.INC'
        include 'tmap_pplv11inc/lunits_inc.decl'
        include 'pplv11inc/LUNITS.INC'
        include 'tmap_pplv11inc/system_inc.decl'
        include 'pplv11inc/SYSTEM.INC'
#else
        INCLUDE 'tmap_pplv11inc:tmap_AXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_MPLOT.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRDL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PLT.INC'
        INCLUDE 'tmap_pplv11inc:tmap_FILNAM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LUNITS.INC'
        INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
        INCLUDE 'PPLV11INC:SYSTEM.INC'
#endif
	IF(IFLG(1).EQ.1)THEN
	    MNX=P(1)
	ELSE
	   MNX=1
	END IF
	IF(IFLG(2).EQ.1.AND.M.GE.2)THEN
	    MNY=P(2)
	ELSE
	    MNY=1
	END IF
	CALL STPRMP('multplt>',8)
	IF(.NOT.QUIETF)WRITE(LTTOUT,
     *	'('' ENTER XLEN FOR COLS 1,2,...,NX'')')
	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	DO 4901 I=1,MNX
	IF(M.GE.I.AND.IFLG(I).EQ.1)THEN
	    XLENM(I)=P(I)
	ELSE
	    XLENM(I)=XLEN
	END IF
4901	CONTINUE
	IF(.NOT.QUIETF)WRITE(LTTOUT,
     *	'('' ENTER YLEN FOR ROWS 1,2,...,NY'')')
	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	DO 4902 I=1,MNY
	IF(M.GE.I.AND.IFLG(I).EQ.1)THEN
	    YLENM(I)=P(I)
	ELSE
	    YLENM(I)=YLEN
	END IF
4902	CONTINUE
	IF(.NOT.QUIETF)WRITE(LTTOUT,'('' ENTER PLOT SPACINGS'')')
	IF(.NOT.QUIETF)WRITE(LTTOUT,
     *	'('' LEFT BNDRY TO COL 1,COL1 TO COL2,ETC...'')')
	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	DO 4903 I=1,MNX
	IF(M.GE.I.AND.IFLG(I).EQ.1)THEN
	    DXM(I)=P(I)
	ELSE IF(I.EQ.1)THEN
	    DXM(1)=1.4
	ELSE
	    DXM(I)=DXM(1)
	END IF
4903	CONTINUE
	IF(.NOT.QUIETF)WRITE(LTTOUT,
     *	'('' ROW1 TO ROW2,...,ROW NY TO BOTTOM'')')
	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	CALL STPRMP('ppl+>',5)
	DO 4904 I=1,MNY
	IF(M.GE.I.AND.IFLG(I).EQ.1)THEN
	    DYM(I)=P(I)
	ELSE IF (I.EQ.1) THEN
	    DYM(1)=1.2
	ELSE
	    DYM(I)=DYM(1)
	END IF
4904	CONTINUE
	HTM=HEIGHT
	WDM=WIDTH
	WIDTH=DXM(1)
	DO 4905 I=1,MNX
4905	WIDTH=WIDTH+XLENM(I)+DXM(I)
	HEIGHT=DYM(MNY)
	DO 4906 I=1,MNY
4906	HEIGHT=HEIGHT+YLENM(I)+DYM(I)
	IF(.NOT.QUIETF)WRITE(LTTOUT,9987) WIDTH,HEIGHT
9987	FORMAT(' PLOT IS',F6.2,' IN. WIDE AND',F6.2,' IN.HIGH')
	IMCNT=0
	IMULT=1
	IXM=0
	IYM=1
#ifdef unix
        open(lmult,status='scratch',form='unformatted')
#else
	OPEN(LMULT,FILE='MULTZZZ.DAT',STATUS='NEW',
     *	FORM='UNFORMATTED')
	INQUIRE(UNIT=LMULT,NAME=MULTFL)
	CLOSE(LMULT)
#endif
10	RETURN
	END

