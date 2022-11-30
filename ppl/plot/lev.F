	SUBROUTINE LEV
C** 
C**    @(#)lev.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INCLUDE 'CMRDL.INC'
	INCLUDE 'parampl5_dat.decl'
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'cont_inc.decl'
	INCLUDE 'CONT.INC'
	INCLUDE 'shade_vars.cmn'


	CHARACTER PRSE*2048

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
*       *jd* rename variable IF to INTIF for MAC 7.92
* 	*jd* 5.6.96 Add code to fix bug in specification of levels
*	where level should be zero but round off error results in near but
*	non zero value.
* v552 *acm* 3/03 Longer strings: changes for multi-line labels.
* v581 *acm* 3/05 open levels (-INF) (INF)
* V600 *acm* 8/05 Fix bug 1335: move the initialization of LWGT(j) = 2 to 
*                 before the possible resetting of zlev if posinf true.
* V603 *acm* 7/07 As done in plotz.F, use TM_FPEQ to compare zlev(i) to 0.,
*                 within machine accuracy.
* V608 *acm*10/07 The above fix refered to TM_FPEQ( ABS(ZLEV(I)) ,0.) ) 
*                 It should be ZLEV(I). Fixes bug 1546.
* V610 *acm* 3/08 - add new MODE NLEVELS; default number of levels is now 40
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V685 *acm* 7/13 Ticket 1990: fixes for number of digits. More precision needed 
*                 (Lavelle examples) 
* v686 *acm* 1/14 Fix ticket 2123: labeling color bar when there is just one level
* v686 *acm* 2/14 Fix ticket 2124: Do the above fix only in the subroutine KEY.
* v692 *acm* 7/14 For centered key labels add a level to corresp to the upper bound
*                 of the top color on thecolorbar, excluding the inf triangle.
* V697 *acm* 12/15 Fix ticket 2326: single level plus (inf) and/or (-inf)
* V741 *acm*  5/18 issue 1874: If they sent in a single level to contour, 
*                 label contour lines with that value, without rounding off
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	DIMENSION VAL(4),INTIF(4)
        INTEGER ntop
        LOGICAL   TM_FPEQ_SNGL

C Initialize

	DO i = 1, ICNLEV
	   ZLEV(i) = 0
	   LDIG(i) = 0
	   LWGT(i) = 0
	   LCLR(i) = 0
	ENDDO

	IF(LABEL.EQ.' ')THEN
	    NLEV=NLEV_DFLT
	    IAUTOC=1
	    GOTO 10
	ENDIF
	CALL UPPER(LABEL,2048)
C
C	BEGIN PARSE
C
        posinf = .FALSE.
        neginf = .FALSE.

        ISTP=-1
	IREL=0
200	LABEL=LABEL(ISTP+2:)
	IST=INDEX(LABEL,'(')+1
	ISTP=INDEX(LABEL,')')-1
	IF(IST.EQ.1)GOTO 10
	IF(ISTP.EQ.-1)THEN
	    CALL CMDERR(10,STRING,ISLEN,LBSTRT+IREL+IST-2)
	    GOTO 10
	ENDIF
	PRSE=LABEL(IST:ISTP)
	DO 11 I=1,4
	  VAL(I)=0.0
	  INTIF(I)=0
11	CONTINUE
	CALL LEVPRS(PRSE,VAL,INTIF,IER,IPOS)
	IERR=LBSTRT+IREL+IST+IPOS-2
	IF(IER.NE.0)THEN
	    CALL CMDERR(IER,STRING,ISLEN,IERR)
	    RETURN
	ENDIF
	IREL=IREL+ISTP+1
	CALL LEVSET(IST,VAL,INTIF,LWGH,NL)
C
C	TEST
C
	IF(LWGH.EQ.0)THEN
	    IF(IAUTOC.EQ.1)NLEV=0
	    IAUTOC=0
C
C	SET CONTOUR LEVELS
C
	    IF(INTIF(1).EQ.0)THEN
		IAUTOC=1
		NLEV=NLEV_DFLT
		GOTO 200
	    ENDIF
	    IF(INTIF(3).EQ.1.AND.VAL(3).EQ.0.)THEN
		DZ=(ZLEV(NLEV)-ZLEV(1))*0.00001
		DO 500 I=1,NL
		Z=VAL(1)+(VAL(2)-VAL(1))*(I-1)
		DO 510 J=1,NLEV
		IF(ABS(Z-ZLEV(J)).LE.DZ)THEN
		    CALL DELLEV(J)
		    GOTO 500
		ENDIF
510		CONTINUE
500		CONTINUE
	    ELSE
C
C	CREATE NEW LEVELS
C
		IF(NLEV+NL.GT.ICNLEV)THEN
		    CALL CMDERR(11,STRING,ISLEN,IERR)
		    RETURN
		ENDIF
		
		DO 100 I=1,NL
		J=NLEV+I

                LWGT(J)=2

                IF (J.GT.1 .AND. posinf) THEN
                   zlev(j) = zlev(j-1) + 1.
                ELSE 
                   ZLEV(J)=VAL(1)+VAL(3)*(I-1)
CC		Bug fix for non zero levels that ought to be zero *jd*
		   IF (ABS(ZLEV(J)) .LT. EPSILON*ABS(VAL(1))) ZLEV(J)=0.0	
                   IF ( TM_FPEQ_SNGL( ABS(ZLEV(J)) ,0.) ) GOTO 100
                ENDIF

		IF(ZLEV(J).GT.0)THEN
		   LWGT(J)=1
	        ELSE
		   LWGT(J)=3
		ENDIF

100		CONTINUE

C
C	FIND MIN AND MAX
C
		ZMIN=ZLEV(1)
                IF (neginf) ZMIN = ZLEV(2)
		ZMAX=ZMIN
                ntop = nlev+nl
                IF (posinf) ntop = ntop - 1
		DO 110 I=2,ntop
		IF(ZLEV(I).GT.ZMAX)ZMAX=ZLEV(I)
		IF(ZLEV(I).LT.ZMIN)ZMIN=ZLEV(I)
110		CONTINUE

                IF (neginf) zlev(1) = zmin - 1.

		DZ=ZMAX-ZMIN
		IF(DZ.EQ.0.)DZ=ABS(ZMAX)
		IF(DZ.EQ.0.)DZ=10000.0
		JJ=NSIGC-IFIX(ALOG10(DZ)+0.5)
		IF(JJ.LE.0) THEN
		   IF (NSIGC .LT. 0) THEN
		     JJ=nsigc+1
		   ELSE
		     JJ = -1
		   ENDIF
		ENDIF
		DO 120 I=1,NL
		J=NLEV+I
		LCLR(J)=0
		IF(INTIF(4).EQ.1)THEN
		    LDIG(J)=VAL(4)
		ELSE
		    IF(ZLEV(J).EQ.0)THEN
			LDIG(J)=-1
		    ELSE
			LDIG(J)=JJ
		    ENDIF
		ENDIF
120		CONTINUE
		NLEV=NLEV+NL
C
C	SORT
C
		CALL LEVSRT
C
C	CHECK FOR DUPLICATE LEVELS?
C
		DZ=DZ*0.00001
		I=1
		IF(NLEV.LE.1)GOTO 330
320		J=I+1
325		IF(ABS(ZLEV(I)-ZLEV(J)).LE.DZ)THEN
		    CALL DELLEV(I)
		ENDIF
		J=J+1
		IF(J.LE.NLEV)GOTO 325
330		I=I+1
		IF(I.LT.NLEV)GOTO 320
	    ENDIF
	ELSE IF(LWGH.EQ.5)THEN  ! DELETE a level
	    DZ=(ZLEV(NLEV)-ZLEV(1))*0.00001
	    DO 600 I=1,NL
	    Z=VAL(1)+VAL(3)*(I-1)
	    DO 610 J=1,NLEV
	    IF(ABS(Z-ZLEV(J)).LE.DZ)THEN
		CALL DELLEV(J)
		GOTO 600
	    ENDIF
610	    CONTINUE
600	    CONTINUE
	ELSE
	    IF(IAUTOC.EQ.1)GOTO 10
C
C	SET WEIGHTS
C
	    DZ=(ZLEV(NLEV)-ZLEV(1))*0.00001
	    DO 400 I=1,NL
	    Z=VAL(1)+VAL(3)*(I-1)
	    DO 410 J=1,NLEV
	    IF(ABS(Z-ZLEV(J)).LE.DZ)THEN
		IF(LWGH.EQ.4)THEN
		    LCLR(J)=VAL(4)
		ELSE
		    LWGT(J)=LWGH
		ENDIF
		GO TO 400
	    ENDIF
410	    CONTINUE
400	    CONTINUE
	ENDIF
	GOTO 200
10	CONTINUE


C issue 1874:
C If they sent in a single level to contour, label with the 
C value without rounding off.  If /SIGDIG was set, apply that setting

	IF (nlev .EQ. 1) THEN
	   
	   vv = INT(ZLEV(1))
	   IF (ZLEV(1) .NE. vv) ldig(1) = 1
	   znum = zlev(1)
	   DO i = 1, 5
	      znum = (10.**FLOAT(i))* ZLEV(1)
	      vv = INT(znum)
	      IF (znum .EQ. vv) EXIT
	      ldig(1) = ldig(1) + 1
	   ENDDO
	   IF (newsig .GT. 0) ldig(1) = nsigc
	ENDIF

	RETURN
	END
