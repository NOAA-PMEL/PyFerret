      SUBROUTINE SYMBOL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
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
C**    @(#)symbol.f	1.1    3/10/88
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
C     CALL SYMBOL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
C
C     THIS SUBROUTINE DRAWS A STRING OF CHARACTERS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X,Y    - COORDINATES OF THE LOWER LEFT-HAND CORNER OF
C                 THE FIRST CHARACTER IN USER'S UNITS.
C        DEGRES - ANGLE AT WHICH THE CHARACTERS ARE TO BE DRAWN
C                 IN DEGREES RELATIVE TO THE X-AXIS.
C        HEIGHT - HEIGHT OF THE CHARACTERS IN INCHES (OR MM).
C        NCHAR  - NUMBER OF CHARACTERS TO BE DRAWN.
C        STRING - ARRAY CONTAINING THE DISPLAY CODES OF THE
C                 CHARACTERS TO BE DRAWN.
C
C     THE TABLES THAT ARE USED TO DEFINE THE SYMBOL CHARACTER SET
C     ARE DESCRIBED BELOW:
C        TAB    - THIS TABLE CONSISTS OF A SERIES OF 6-BIT CODES
C                 WHICH CONTAIN THE ACTUAL CHARACTER DEFINITIONS.
C                 WHEN A CODE IS DIVIDED BY 7, THE TRUNCATED QUO-
C                 TIENT IS THE VERTICAL DISPLACEMENT FROM THE
C                 REFERENCE POSITION AND THE REMAINDER IS THE
C                 HORIZONTAL DISPLACEMENT.  A CODE OF 63 INDI-
C                 CATES THAT THE PEN IS TO BE RAISED.
C        PNTTAB - THIS TABLE CONTAINS POINTERS TO THE START OF
C                 EACH OF THE CHARACTER DEFINITIONS AND OTHER
C                 INFORMATION NEEDED TO DRAW THE CHARACTERS.
C                    I = 4 * CHARACTER DISPLAY CODE + 1
C                    PNTTAB(I) = WIDTH OF THE CHARACTER (BITS
C                       1-3)
C                              = VERTICAL REFERENCE TRANSLATION
C                       (BITS 4-6)
C                    PNTTAB(I+1) = NUMBER OF CODES REQUIRED TO
C                       DEFINE THE CHARACTER.
C                    PNTTAB(I+2)+PNTTAB(I+3) = POINTER TO THE
C                       FIRST CHARACTER DEFINITION CODE.
C
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
      LOGICAL SYMROT
C
C	RSX-11M
C
C	LOGICAL*1 STRING(120),ASCII(128),NUM(64)
C
C	END RSX-11M
C
C	VAX/VMS
C
	CHARACTER STRING*(*)
	INTEGER*2 ASCII(128),NUM(64)
C
C	END VAX/VMS
C
	INTEGER*2 TAB(2,477),TAB1(528),TAB2(426)
	INTEGER*2 IPNT(64),WDTH(64)

C	LINUX PORT
	integer*4	i4_wdth

C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	EQUIVALENCE (TAB1(1),TAB(1,1)),(TAB2(1),TAB(1,265))
C
	DATA WDTH/5,9,9,9,9,9,9,9,9,7,9,9,9,9,
     *	9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,
     *	9,9,9,9,9,9,9,7,6,6,9,9,9,5,5,6,5,5,9,
     *	9,9,7,9,9,9,9,9,9,9,7,5/
	DATA NUM/11,8,11,8,7,7,6,10,8,8,6,8,3,5,
     *	6,9,7,12,11,12,5,6,5,5,9,6,6,9,6,9,11,4,
     *	10,12,4,16,12,5,2,11,2,4,4,18,5,0,8,5,3,
     *	4,4,22,7,6,12,11,6,6,3,3,6,6,2,14/
	DATA IPNT/1,12,20,31,39,46,53,59,69,77,85,
     *	91,99,102,107,113,122,129,141,152,164,169,
     *	175,180,185,194,200,206,215,221,230,241,245,
     *	255,267,271,287,299,304,306,317,319,323,327,
     *	345,0,350,358,363,366,370,374,396,403,409,421,
     *	432,438,444,447,450,456,462,464/
	DATA ASCII/32*46,55,53,61,44,52,62,49,42,43,
     *	40,38,47,39,48,41,28,29,30,31,32,33,34,35,
     *	36,37,1,64,59,45,60,56,46,2,3,4,5,6,7,8,9,10,
     *	11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
     *	26,27,50,63,51,57,54,58,2,3,4,5,6,7,8,9,10,11,
     *	12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
     *	27,6*46/
     	 DATA TAB1/0,13,1,14,2,13,1,12,0,13,10,0,0,17,1,18,2,17,1,16,
     *	0,17,0,0,0,5,3,8,6,5,6,0,10,0,0,3,6,3,5,4,6,3,6,1,5,0,0,0,0,8,
     * 5,8,6,7,6,5,5,4,0,4,6,2,4,0,2,0,0,2,0,6,2,8,4,8,6,6,4,8,0,8,
     *	0,0,4,0,6,2,6,6,4,8,0,4,4,4,10,0,6,8,0,8,0,0,6,0,0,4,4,4,
     *	10,0,6,8,0,8,0,0,3,3,6,3,6,2,4,0,2,0,0,2,0,6,2,8,4,8,6,6,
     *	0,0,0,8,10,0,0,4,6,4,10,0,6,8,6,0,0,0,4,0,10,0,2,0,2,8,10,0,
     *	0,8,4,8,0,3,0,2,2,0,4,0,6,2,6,8,0,8,0,0,10,0,0,2,6,8,10,0,
     *	2,4,6,0,0,8,0,0,6,0,0,0,0,8,3,2,6,8,6,0,0,0,0,8,1,8,5,0,
     *	6,0,6,8,4,0,6,2,6,6,4,8,2,8,0,6,0,2,2,0,4,0,0,0,0,8,5,8,
     *	6,7,6,5,5,4,0,4,4,0,6,2,6,6,4,8,2,8,0,6,0,2,2,0,4,0,10,0,
     *	4,2,6,0,0,0,0,8,5,8,6,7,6,5,5,4,0,4,10,0,3,4,6,1,6,0,0,1,
     *	1,0,5,0,6,1,6,3,5,4,1,4,0,5,0,7,1,8,5,8,6,7,3,0,3,8,10,0,
     *	0,8,6,8,0,8,0,2,2,0,4,0,6,2,6,8,0,8,0,6,3,0,6,6,6,8,0,8,
     *	0,0,3,3,6,0,6,8,0,8,0,7,6,1,6,0,10,0,0,0,0,1,6,7,6,8,0,8,
     *	3,5,6,8,10,0,3,5,3,0,0,8,6,8,6,7,0,1,0,0,6,0,3,0,4,1,4,7,
     *	3,8,1,8,0,7,0,1,1,0,3,0,0,6,2,8,2,0,10,0,0,0,4,0,0,6,0,7,
     *	1,8,5,8,6,7,6,6,0,1,0,0,6,0,0,1,1,0,5,0,6,1,6,4,5,5,2,5,
     *	10,0,3,5,6,8,0,8,5,0,5,8,0,3,6,3,0,1,1,0,5,0,6,1,6,4,5,5,
     *	1,5,0,4,0,8,6,8,0,3,1,4,5,4,6,3,6,1,5,0,1,0,0,1,0,7,1,8/
	DATA TAB2/
     *	5,8,6,7,0,7,0,8,6,8,2,0,1,4,0,3,0,1,1,0,5,0,6,1,6,3,5,4,
     *	1,4,0,5,0,7,1,8,5,8,6,7,6,5,5,4,0,1,1,0,5,0,6,1,6,7,5,8,
     *	1,8,0,7,0,5,1,4,5,4,6,5,0,4,6,4,10,0,3,7,3,1,0,4,6,4,0,4,
     *	6,4,10,0,3,7,3,1,10,0,1,2,5,6,10,0,1,6,5,2,0,0,4,8,2,0,0,2,
     *	0,6,2,8,0,0,2,2,2,6,0,8,1,2,2,1,4,1,5,2,5,3,4,4,2,4,1,5,
     *	1,6,2,7,4,7,5,6,10,0,4,8,4,0,10,0,2,0,2,8,0,2,6,2,10,0,0,5,
     *	6,5,0,10,1,10,2,11,2,13,1,14,0,13,1,12,2,13,0,13,1,14,
     *	2,13,1,12,0,13,1,9,2,9,1,7,2,0,0,0,0,8,2,8,
     *	0,0,2,0,2,8,0,8,0,6,0,7,1,8,2,8,3,7,3,6,2,5,1,5,0,6,10,0,
     *	0,1,6,7,10,0,6,2,6,1,5,0,4,0,3,1,3,2,4,3,5,3,6,2,1,9,2,9,
     *	1,7,10,0,4,9,5,9,4,7,3,1,0,4,3,7,10,0,0,4,6,4,2,3,3,5,3,8,
     *	1,8,1,5,2,3,10,0,2,2,3,1,2,0,1,1,2,2,1,6,3,8,5,6,3,4,3,3,
     *	10,0,3,2,4,1,3,0,2,1,3,2,0,4,3,7,6,4,10,0,3,7,3,0,3,7,3,0,
     *	3,0,0,3,3,0,6,3,6,1,0,4,6,7,0,7,6,4,0,1,0,0,6,0,10,0,6,1,
     *	0,4,6,7,0,7,6,4,0,1,10,0,0,0,6,0,0,8,4,0,0,17,1,18,2,17,
     *	1,16,0,17,10,0,0,10,1,10,2,11,2,13,1,14,0,13,1,12,2,13/
C
C     DEFINE THE SCALE FACTOR TO BE THE AMOUNT OF MAGNIFICATION
C     REQUIRED TO CONVERT THE CHARACTERS IN THE CHARACTER DEFI-
C     NITION TABLE TO THE SIZE SPECIFIED BY THE USER.
C
      SYMSC = HEIGHT * FACTOR / 8.0
100	SYMROT = .FALSE.
C
C     ROTATE THE CHARACTER STRING.
C
      IF(DEGRES .NE. 0.0)THEN
C
C     SET THE CHARACTER ROTATION FLAG.
C
        SYMROT = .TRUE.
C
C     CONVERT THE ROTATION TO RADIANS.
C
        TEMP = 0.01745329251994 * DEGRES
C
C     COMPUTE THE SINE AND COSINE.
C
        SYMSIN = SIN(TEMP)
        SYMCOS = COS(TEMP)
      ENDIF
C
C     SET VECTORS MODE.
C
      PMODE = VECTOR
C
C     START AT THE LOWER LEFT-HAND CORNER OF THE FIRST CHARACTER.
C
      XTEMP = X * XSCALE
      YTEMP = Y * YSCALE
      PENF = .FALSE.
      CALL XYZPLT
      IF(NCHAR .LE. 0)RETURN
      XORG = XTEMP
      YORG = YTEMP
      XREF = 0.0
C
C     DRAW THE CHARACTERS IN THE STRING ONE BY ONE.
C
      DO 20 I=1,NCHAR
C
C     COMPUTE THE POINTER TABLE INDEX.
C
C
C     GET CONTROL INFORMATION FROM THE POINTER TABLE.
C
C
C	RSX-11M
C
C	ICH=ASCII(STRING(I))
C
C	END RSX-11M
C
C	VAX/VMS
C
	ICH=ASCII(ICHAR(STRING(I:I)))
C
C	END VAX/VMS
C
	IPOINT=IPNT(ICH)-1

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
*ORIG	WIDTH=FLOAT(WDTH(ICH)) -- LINUX PORT...
	i4_wdth = wdth(ich)
	WIDTH=FLOAT(i4_wdth)

	NUMCRD=NUM(ICH)
      IF(NUMCRD .NE. 0)THEN
        PENF = .FALSE.
C
C     DRAW THE CHARACTER.
C
        DO 10 J=1,NUMCRD
C
C     GET A CHARACTER DEFINITION CODE FROM THE TABLE.
C
	ICHX=TAB(1,IPOINT+J)
	ICHY=TAB(2,IPOINT+J)
C
C     CHECK FOR THE CHARACTER THAT INDICATES THE PEN IS TO BE
C     RAISED.
C
        IF(ICHX .EQ. 10)THEN
          PENF = .FALSE.
        ELSE
C
C     CALCULATE THE NEW PEN COORDINATES RELATIVE TO THE LOWER
C     LEFT-HAND CORNER OF THE FIRST CHARACTER.
C
          XSYM = XREF + FLOAT(ICHX)
	IF(ICHY.GE.10)THEN
	YSYM=FLOAT(ICHY-12)
	ELSE
	YSYM=FLOAT(ICHY)
	ENDIF
C
C     SCALE AND ROTATE THE NEW COORDINATES.
C
          IF(SYMROT)THEN
            XTEMP = (XSYM * SYMCOS - YSYM * SYMSIN) * SYMSC + XORG
            YTEMP = (XSYM * SYMSIN + YSYM * SYMCOS) * SYMSC + YORG
          ELSE
            XTEMP = XSYM * SYMSC + XORG
            YTEMP = YSYM * SYMSC + YORG
          ENDIF
C
C     MOVE THE PEN TO THE NEW COORDINATES.
C
          CALL XYZPLT
          PENF = .TRUE.
        ENDIF
   10   CONTINUE
      ENDIF
C
C     ADJUST THE REFERENCE POSITION TO TAKE INTO ACCOUNT THE
C     CHARACTER THAT WAS JUST DRAWN.
C
      XREF = XREF + WIDTH
   20 CONTINUE
C
C     SCALE AND ROTATE THE FINAL COORDINATES.
C
      IF(SYMROT)THEN
        XTEMP = XREF * SYMCOS * SYMSC + XORG
        YTEMP = XREF * SYMSIN * SYMSC + YORG
      ELSE
        XTEMP = XREF * SYMSC + XORG
        YTEMP = YORG
      ENDIF
C
C     PLOT TO THE END OF THE CHARACTER STRING.
C
      PENF = .FALSE.
      CALL XYZPLT
      RETURN
C
	ENTRY SYMBL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
	SYMSC=HEIGHT*YSCALE/8.0
	GOTO 100
      END
