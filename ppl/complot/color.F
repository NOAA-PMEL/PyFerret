      SUBROUTINE COLOR(ICODE)
C**
C**    @(#)color.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     CALL COLOR(ICODE)
C
C     THIS SUBROUTINE SELECTS A PEN ON THE TEKTRONIX 4663 PLOTTER,
C     THE HP PLOTTER, AND THE ZETA PLOTTER AND CONTROLS THE SIX-
C     POSITION TURRET HEAD ON THE GERBER PLOTTER.  IT CAN BE USED
C     TO CHANGE THE COLOR OF THE INK USED FOR PLOTTING.
C
C
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
*	Modified J Davison JISAO/PMEL/TMAP 3.23.89
*	To accomodate GKS changes to bundled line attributes
 
	integer	icode

#ifdef unix
        INTEGER TEKCLR(6),TK41XX(3)
        INTEGER NUM(0:15),setlineindex
#else
	byte TEKCLR(6),TK41XX(3)
        byte NUM(0:15)
#endif
	character*1 cnum(0:8)
C
#ifdef unix
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'pplv11inc/PLTCOM.DAT'
	include 'tmap_pplv11inc/gkscm1_inc.decl'
	include 'pplv11inc/GKSCM1.INC'
	include 'tmap_pplv11inc/pltl_inc.decl'
	include 'pplv11inc/PLTL.INC'

#else
	INCLUDE 'tmap_pplv11inc:tmap_PLTCOM.DAT'
	INCLUDE 'tmap_pplv11inc:tmap_GKSCM1.INC'
	INCLUDE 'tmap_pplv11inc:tmap_pltl.INC'
#endif

C
	data cnum/'0','1','2','3','4','5','6','7','8'/
        DATA NUM/48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63/
c	DATA TEKCLR/'`','h','e','m','f','n'/,TK41XX/27,77,76/
	DATA TEKCLR/96,104,101,109,102,110/,TK41XX/27,77,76/
C	DATA TEKPEN/27,67,66,80/
C
C     CHECK THE PLOT TYPE.
C
       GO TO(10,50,40,10,10,20,20),PTYPE + 3
C	GOTO (50,50,40,30,40),PTYPE + 3
C
10	IF(TTYPE.LT.0)GOTO 100
	COLOUR=ICODE
	IF(TTYPE.LT.4100)THEN
	IF(COLOUR.LT.1.OR.COLOUR.GT.6)COLOUR=1
C
C	SEND SEQUENCE FOR TEK 4014 TERMINAL
C
	CALL CHOUT(ESC,1)
	CALL CHOUT(TEKCLR(COLOUR),1)
	ELSE
	IF(COLOUR.LT.0.OR.COLOUR.GT.15)COLOUR=15
C
C	SEND SEQUENCE FOR TEK 41XX TERMINALS
C

CC	Mod for linux *jd* 12.96  
CC	"Inconsistent structure for arg 1 in call to CHOUT" compiler error
CC	as CHOUT is called with arg 1 a scalar, and also an array.
CC	Will commence using the compile option -mismatch, where 
CC	match of subroutine arguments is not forced.


	CALL CHOUT(TK41XX,3)
	CALL CHOUT(NUM(COLOUR),1)
	ENDIF	
C
C     CHECK THE TEKTERMINAL TYPE.
C
C   10 IF(TTYPE .EQ. -4663)THEN
C        COLOUR = ICODE
C
C     CHECK FOR A LEGAL PEN COLOR CODE.
C
C        IF(COLOUR .LT. 1 .OR. COLOUR .GT. 2)COLOUR = 1
C
C     CHECK THE GRAPHICS FLAG.
C
C        IF(.NOT. GRAPHF)THEN
C
C     TRANSMIT THE 'PLOTTER ON' COMMAND TO THE TEKTRONIX PLOTTER.
C
C	CALL (TEKON,3)
C        ENDIF
C
C     SELECT THE APPROPRIATE PEN.
C
C	CALL CHOUT(TEKPEN,4)
C        CALL CHOUT(COLOUR + 48,1)
C
C     CHECK THE GRAPHICS FLAG.
C
C        IF(.NOT. GRAPHF)THEN
C
C     TRANSMIT THE 'PLOTTER OFF' COMMAND TO THE TEKTRONIX PLOTTER.
C
C	CALL CHOUT(TEKOFF,5)
C
C     WRITE OUT THE CHARACTER BUFFER.
C
C          CALL CHDMP
C        ENDIF
C      ENDIF
C     CHECK FOR OTHER PLOTTING DEVICES.
C
	GOTO 100
C
C	GKS COLOR AND SETUP
C
20	COLOUR=ICODE
	IF(.NOT.GKSOPN)THEN
#ifdef core
            call crinit
#else
	    call gkinit
	    CALL GCLRWK(WSID,0)
#endif
C
	    GKSOPN=.TRUE.
	ENDIF
	CALL PPLGFLUSH
#ifdef core
        istat=setlineindex(icode)
#else
*	CALL GSPLCI(ICODE) ! original
#endif

1000	call gspli (max(icode,1)) ! JD 3.23.89 & 3.13.90

100	GOTO(50,30,30,30,40,30,40),PTYPE + 3
   40  COLOUR = ICODE
	IF(.NOT.PLTFLG)THEN
C	INITIALIZE THE BINARY BUFFER WITH WIDTH,HEIGHT
C	OF THE PLOTTING AREA
C
	CALL BINBUF(ASIZE,BSIZE)
	PLTFLG=.TRUE.
	ENDIF
C
C	SELECT THE PEN
C
	CALL BINBUF(FLOAT(COLOUR),-1.)
	PEN=.FALSE.
	LPEN=.NOT.PEN
      GOTO 30
C
50	COLOUR=ICODE
C
C	COLOR ON THE HP7475
C
	IF(COLOUR.LT.1.OR.COLOUR.GT.8)COLOUR=1
	IF(.NOT.PLTFLG)THEN
	CALL ZBUFFT('DF;',3)
	PLTFLG=.TRUE.
	ENDIF
	CALL ZBUFFT('PU;SP',5)
	CALL ZBUFFT(CNUM(COLOUR),1)
	CALL ZBUFFT(';',1)
	PEN=.FALSE.
	LPEN=.NOT.PEN
30	RETURN
      END
