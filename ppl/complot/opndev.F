	subroutine opndev(termf,clear)
C**
C**    @(#)opndev.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
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
* revised for TMAP
* from PPLUS :
*       Directory PMEL:[PLOT5.PPLV11.COMPLOT]OPNDEV.F -->.FOR via "PREPRS"
* search for "TMAP" to find all revisions
* *sh* 8/26/88 - rev 1.0 - for FERRET 1.10+
* *sh* 1/20/89 - rev 1.1 - for FERRET 1.10+ from V11 of PPLUS
*       - only delete segments currently in use to permit multiple viewports
* *jd* 3.20.89 Modified for GKS metafiles to write private GCLRWK to metafile
*	as metafile system doesn't write it for some unknown reason.
* *jd* 7.12.90 Mod to not call CHOUT if dimension arg eq 0
* *jd* 10.26.93 Mod to eliminate private GCLRWK for xgks
* *jd* 12.96 Numerous comments including control characters of various
* 	nature (eg ^) eliminated.  They confuse the linux F90 compiler


*****************************************************

C
C	CALL OPNDEV(TERMF,CLEAR)
C
C	THIS SUBROUTINE IS USED TO OPEN A DEVICE FOR THE FIRST TIME
C	AND TO OPEN THE DEVICE FOR ADDITIONAL INPUT.
C
C	TERMF  -- FOR INTERACTIVE / COMMAND SWITCH
C		 IF TRUE THE COMMANDS ARE COMING FROM THE TERMINAL
C	CLEAR -- LOGICAL IF TRUE CLEAR THE DEVICE  (I.E. START NEW)
C
C
* TMAP mod 8/26/88
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. Remove obsolete ifdefs

	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'
	include 'gkscm2.cmn' !jd 3.20.89 for private gclrwk

	INTEGER	iseg, ivp, lens, itype, i, j, nterm, ndelay ! local declarations
	integer	control_flag ! For private GCLRWK

* special COMMON to let this routine know if FERRET is in control


	include 'ppl_in_ferret.cmn'

* end of TMAP mods 8/26/88 & 1/20/89
C
C	THIS IS A VAX/VMS SUBROUTINE AT THIS TIME
C
	LOGICAL CLEAR,TERMF
	LOGICAL FRST,TRMOPN
	integer*4 lib$get_symbol,len,stat
	byte FIRSTB(20,12),SECNDB(20,12),FIRST(20),SECND(20)
	byte ER4010(2),ER41XX(8)
	CHARACTER TERM(12)*10,TYPE*10
	INTEGER ITERM(12),FIRSTL(12),SECNDL(12)
	LOGICAL SEGMNL(12),SEGMN,FCLOSL(12),FCLOS
C
C
	DATA ER4010/27,12/,ER41XX/27,83,75,33,27,83,69,49/
	DATA TERM/'VT240','GVT+','ZENITH','TEK4010','TEK41XX','MAC',
     *		  'TAB','TEK4105','TEK4014','HIREZ','HP2397','GP220'/
	DATA ITERM/1,2,3,4,5,4,6,7,4,8,9,10/,FRST/.TRUE./,NTERM/12/
C
C
C
	DATA FIRSTB/27,91,63,51,56,104,14*0,
     *		    28,19*0,
     *		    27,91,50,74,16*0,
     *		    20*0,
     *		    27,37,33,48,27,76,86,48,27,83,86,33,49,7*0,
     *		    29,19*0,
     *		    27,37,33,48,27,76,86,48,12*0,
     *		    27,91,50,74,27,49,14*0,
     *		    27,42,100,102,27,42,100,99,27,91,51,56,104,7*0,
     *		    29,19*0,
     *		    20*0,
     *              20*0/
	DATA SECNDB/27,91,63,51,56,108,14*0,
     *		    24,19*0,
     *		    27,58,56,102,16*0,
     *		    20*0,
     *		    27,83,78,27,83,86,33,48,27,76,86,49,27,37,33,49,4*0,
     *		    27,34,48,103,16*0,
     *		    27,76,86,49,27,37,33,49,12*0,
     *		    27,50,18*0,
     *      27,91,51,56,108,27,42,100,100,27,42,100,101,7*0,
     *		    2,19*0,
     *		    20*0,
     *              20*0/
C
C	DEVICE HAS SEGMNMENTS IF TRUE  (I.E. 4115, 4107)
C
	DATA SEGMNL/ .FALSE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		    .TRUE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		   .FALSE.,
     *		 2*.FALSE./
C
C	DEVICE MUST BE ERASED TO GO BACK TO COMMAND
C
	DATA FCLOSL/ .TRUE.,
     *		    .FALSE.,
     *		     .TRUE.,
     *		     .TRUE.,
     *		    .FALSE.,
     *		     .TRUE.,
     *		    .FALSE.,
     *		     .TRUE.,
     *		    .FALSE.,
     *		    .FALSE.,
     *		  2*.FALSE./
	DATA FIRSTL/6,1,4,0,13,1,8,6,13,1,2*0/
	DATA SECNDL/6,1,4,0,16,4,8,2,13,1,2*0/
C
C	SET UP FOR SPECIFIC TERMINAL TYPE
C
	IF(FRST)THEN
            call getenv('GRAPHTERM',TYPE)
            LENS=lnblk(type,10)
            IF(lens.le.0)then
		ITYPE=1
	    ELSE
		DO 14 I=1,NTERM
		   IF(TYPE(1:LENS).EQ.TERM(I))GOTO 11
14		CONTINUE
		I=1
11		ITYPE=ITERM(I)
	    ENDIF
	    DO 12 I=1,20
	      FIRST(I)=FIRSTB(I,ITYPE)
12	    CONTINUE
	    DO 13 I=1,20
	      SECND(I)=SECNDB(I,ITYPE)
13	    CONTINUE
	    FRST=.FALSE.
	    SEGMN=SEGMNL(ITYPE)
	    FCLOS=FCLOSL(ITYPE)
	ENDIF
C
C
	PMODE = ALPHA
	IF(SAVEF)CALL XYZPLT
C
C	CHECK FOR PLOT-TYPE
C
	IF(PTYPE.EQ.1.OR.PTYPE.EQ.2)THEN
C
C	DEVICE IS A TEK TERMINAL COMPATIBLE
C
	    IF(.NOT.TRMOPN)THEN
C
C	OPEN DEVICE FOR GRAPHICS AND ERASE THE SCREEN
C
		if (firstl(itype) .gt. 0) CALL CHOUT(FIRST,FIRSTL(ITYPE))
		TRMOPN=.TRUE.
	    ENDIF
	    IF(CLEAR)THEN
		IF(SEGMN)THEN
		    CALL CHOUT(ER41XX,8)
		ELSE
	            CALL CHOUT(ER4010,2)
C
C	TRANSMIT SYNC CHARACTERS FOR PHOSPHORE COOL DOWN.
C
c		    NDELAY=IFIX(0.7*FLOAT(BAUDR/10))
c		    DO 30 J=1,NDELAY
c30		    CALL CHOUT(SYN,1)
		ENDIF
	    ENDIF
	    CALL CHDMP
	ELSE IF(PTYPE.EQ.3.OR.PTYPE.EQ.4)THEN
C
C	DEVICE IS A GKS DEVICE
C
	    CALL PPLGFLUSH

	    IF(CLEAR) THEN
	       IF ( PPL_in_FERRET ) THEN
	         CALL CLEAR_VP
	       ELSE
		 control_flag = 0
	         CALL FGD_GCLRWK(WSID,control_flag)
	       ENDIF
	    ENDIF

	ENDIF
	IF(PTYPE.EQ.0.OR.PTYPE.EQ.2.OR.PTYPE.EQ.4)THEN
C
C	THE OUTPUT IS A BINARY FILE
C
	    IF(PLTFLG)THEN
C
C	MOVE TO A NEW PLOTTING AREA
C
		PEN = .FALSE.
		CALL ZABMV
		IF(CLEAR)THEN
		    CALL BINFSH
		    PLTFLG = .FALSE.
		ENDIF
	    ENDIF
	ENDIF
	LPEN = .FALSE.
C
C     INITIALIZE THE DATA VARIABLES FOR THE NEXT PLOT.
C
	A = 0.0
	B = 0.0
	AOLD = 0.0
	BOLD = 0.0
	COLD = 0.0
C
	IF(TTYPE.GT.0)THEN
	    IF(ASIZE / TWIDTH .GT. BSIZE / THIGHT)THEN
		PLTOTA = TSCLA /ASIZE
		PLTOTB =TSCLB /ASIZE * TWIDTH /THIGHT
	    ELSE
		PLTOTA = TSCLA / BSIZE * THIGHT / TWIDTH
		PLTOTB = TSCLB / BSIZE
	    ENDIF
	ELSE
	    PLTOTA=TSCLA/(TWIDTH*FACTOR)
	    PLTOTB=TSCLB/(THIGHT*FACTOR)
	ENDIF
C
C     RECOMPUTE THE HP PLOTTER CONVERSION FACTORS.
C
C
C     RESTORE THE DEFAULT MINIMUM AND MAXIMUM ALLOWABLE VALUES OF
C     A AND B.
C
	AMIN = 0.0
	BMIN = 0.0
	AMAX = ASIZE
	BMAX = BSIZE
C
C     CALCULATE THE OFF-SCREEN FLAG.
C
	LOSFLG = 0
	IF(AOLD - 0.001 .GT. AMAX)LOSFLG = 1
	IF(AOLD + 0.001 .LT. AMIN)LOSFLG = 2
	IF(BOLD - 0.001 .GT. BMAX)LOSFLG = LOSFLG + 4
	IF(BOLD + 0.001 .LT. BMIN)LOSFLG = LOSFLG + 8
C
C     TRANSFORM THE ABSOLUTE ORIGIN TO THE USER'S COORDINATE SYS-
C     TEM.
C
	CALL TFORMI(XOLD,YOLD,ZOLD)
	IF(WINDOF)THEN
C
C     CALCULATE THE OFF-WINDOW FLAG.
C
	    LOWFLG = 0
	    IF(XOLD .GT. XMAX)LOWFLG = 1
	    IF(XOLD .LT. XMIN)LOWFLG = 2
	    IF(YOLD .GT. YMAX)LOWFLG = LOWFLG + 4
	    IF(YOLD .LT. YMIN)LOWFLG = LOWFLG + 8
	ENDIF
	RETURN
C
	ENTRY CLSDEV(TERMF,CLEAR)
C
C	THIS ROUTINE CLOSES THE DEVICE DEPENDING ON THE SITUATION
C
	IF(PTYPE.EQ.1.OR.PTYPE.EQ.2)THEN
C
C	DEVICE IS TEK TERMINAL COMPATIBLE
C
	    IF(FCLOS.OR.CLEAR)THEN
C
C	DEVICE REQUIRES THE TERMINAL CLEARED
C
		CALL CHOUT(ER4010,2)

	    ENDIF
C
C	GO TO COMMAND SCREEN
C
	    if (secndl (itype) .gt. 0) CALL CHOUT(SECND,SECNDL(ITYPE))
	    TRMOPN=.FALSE.
	    CALL CHDMP
	ELSE IF(PTYPE.EQ.3.OR.PTYPE.EQ.4)THEN
C
C	DEVICE IS GKS
C
	    CALL PPLGFLUSH
	ELSE IF((PTYPE.EQ.0.OR.PTYPE.EQ.2.OR.PTYPE.EQ.4)
     *		.AND.CLEAR)THEN
	    IF(PLTFLG)THEN
		PEN=.FALSE.
		CALL ZABMV
		CALL BINFSH
		PLTFLG=.FALSE.
	    ENDIF
	ENDIF
	RETURN
C
	END
