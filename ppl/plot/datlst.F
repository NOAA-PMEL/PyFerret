      SUBROUTINE DATLST(LLUN,X,NSIZE,NXS,NYS) 
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
C**    @(#)datlst.F	1.1    3/10/88
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	DIMENSION X(NSIZE)

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'LINES.INC'
	INCLUDE 'HD.INC'
	INCLUDE 'CMRD.INC'

	IY=NSIZE/2
	NXL=1 
	NYL=1 
	IF(M.GE.1.AND.IFLG(1).EQ.1)NXL=IFIX(P(1)) 
	IF(M.GE.3.AND.IFLG(3).EQ.1)NYL=IFIX(P(3)) 
	IF(JTYPE.LE.0.OR.JTYPE.EQ.2)THEN 
	    NXH=NX 
	    NYH=NY 
	    IF(M.GE.2.AND.IFLG(2).EQ.1)NXH=IFIX(P(2)) 
	    IF(M.GE.4.AND.IFLG(4).EQ.1)NYH=IFIX(P(4)) 
	    IF(JTYPE.EQ.-1.OR.JTYPE.EQ.2)THEN
		IF(M.GE.5.AND.IFLG(5).EQ.1.AND.P(5).EQ.2)THEN
		    WRITE(LLUN,9994)'SECOND'
9994		    FORMAT(9X,A6,' COMPONENT')
		    IVLEN=NX*NY
 	ELSE
		    WRITE(LLUN,9994)'FIRST'
		    IVLEN=0
		ENDIF
	    ELSE
		IVLEN=0
	    ENDIF
	    K=(NXH-NXL)/7+1 
	    DO 100 K1=1,K 
	    K8=(K1-1)*7+(NXL-1) 
	    IM=NXH-K8 
	    IF(IM.GT.7)IM=7 
	    WRITE(LLUN,9999)(I+K8,I=1,IM) 
9999	    FORMAT(9X,7I10) 
	    DO 101 J=NYL,NYH 
	    WRITE(LLUN,9998)J,(X(IVLEN+NXS*(J-1)+I+K8),I=1,IM) 
101	    CONTINUE
9998	    FORMAT(6X,I6,1P7E10.2) 
	    WRITE(LLUN,'(/)') 
100	    CONTINUE
	ELSE 
	    IBASE=1 
	    NXH=32000 
	    NYH=LNUM 
	    IF(M.GE.2.AND.IFLG(2).EQ.1)NXH=IFIX(P(2)) 
	    IF(M.GE.4.AND.IFLG(4).EQ.1)NYH=IFIX(P(4)) 
	    DO 300 J=1,NYL-1 
	      IBASE=IBASE+LLENG(J) 
300	    CONTINUE
	    DO 200 J=NYL,NYH 
	    WRITE(LLUN,9997)J,LLENG(J) 
9997	    FORMAT(//9X,'LINE=',I4,'  LENGTH=',I6) 
	    WRITE(LLUN,9996) 
9996	    FORMAT(11X,'I',10X,'X',15X,'Y') 
	    IB=IBASE+LLENG(J)-1 
	    IF(IB.GT.IBASE+NXH-1)IB=IBASE+NXH-1 
	    WRITE(LLUN,9995)(I-IBASE+1,X(I),X(IY+I),I=IBASE+NXL-1,IB) 
9995	    FORMAT(6X,I6,1P2E15.4) 
	    IBASE=IBASE+LLENG(J) 
200	    CONTINUE 
	END IF 
	RETURN
	END 
