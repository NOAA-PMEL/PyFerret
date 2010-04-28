	SUBROUTINE LEVEL
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
C**    @(#)level.F	1.1    3/10/88
C** V600:  1/06 *acm*  prevent values of LOG10(0.0)
* V610 *acm* 3/08 - add new MODE NLEVELS; default number of levels is now 40
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'CMRD.INC'
	INCLUDE 'cont_inc.decl'
	INCLUDE 'CONT.INC'

	DO 2601 I=1,M 
	IF(IFLG(I).NE.0)ZLEV(I)=P(I) 
2601	CONTINUE 
	NLEV=M 
	IF(M.LE.1)THEN 
	    NLEV=IFIX(P(1)) 
	    IF(M.EQ.0)NLEV=NLEV_DFLT
	    IAUTOC=1 
	    GOTO 10 
	ELSE 
	    IAUTOC=0 
            IF (ZLEV(1)-ZLEV(NLEV) .EQ. 0) THEN
	       JJ=NSIGC
            ELSE 
	       JJ=NSIGC-IFIX(ALOG10(ABS(ZLEV(1)-ZLEV(NLEV)))+.5) 
	    ENDIF
            IF(JJ.LE.0)JJ=-1 
	    DO 2602 I=1,NLEV 
	    LCLR(I)=0
	    IF(ZLEV(I).EQ.0.)THEN 
	      LDIG(I)=-1 
	    ELSE 
		LDIG(I)=JJ
	    END IF
2602	    CONTINUE
	END IF 
	IFLG(1)=0 
	DO 2603 I=1,NLEV 
	LWGT(I)=2 
	IF(ABS(ZLEV(I)).LT.ABS(ZLEV(1)-ZLEV(2))*1.E-5)GOTO 2603 
	IF(ZLEV(I).GT.0.)THEN 
	    LWGT(I)=1 
	ELSE 
	    LWGT(I)=3 
	END IF 
2603	CONTINUE 
10	RETURN
	END
