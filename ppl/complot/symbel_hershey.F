 	SUBROUTINE SYMBEL_HERSHEY(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
C** 
C**    @(#)symbel.tplate.F	1.1    3/10/88
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
C       Mod *JD* 3.19.91 for unix -- use environment variable PLOTFONTS
C       *jd* 7.92 variable IF renamed to KIF for MACMAC
*       *jd* 11.94 

*	Mod *jd* 11.30.95 to fix ragged symbols to to precision loss
*
*       Mod *kob* 1.28.97 - Reorganization of the original code to remove
*			    an illegal (f90) fortran forward reference into
*			    the middle of a do loop.  This change was 
*		            necessary to get this routine to compile on linux
*			    and also to fix a plotting bug with the new 
*			    solaris version (v4.0) of f77.  This change has 
*			    occurred in the middle of releasing v4.45 of 
*		  	    Ferret to the world.  Aix, Sgi, HP, OSF and SunOs
*			    have already been released, and this change is
*		            not a part of them.  This change is needed
*			    to release linux and more importantly solaris.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C *ACM* 6/02   When there is a period or decimal point in the text, make
C              a GKS dot there too, so it'll show up in small labels.
C *acm* 7/02   Take out the MARKER calls that made dots. This caused trouble 
C              when making encapsulated postscript from postscript.  
C *acm* 10/14  Ticket 2203: on the 64-bit machines, the \\ for backspace
C              should be \ .
C *kms* 06-2017 - change from PLOTFONTS to FER_FONTS
C *acm* 07-2017 - ticket 2556 Look for FER_FONTS if not defined also check PLOTFONTS C
** 10/2022 *acm* Code cleanup: lengths of strings not to overflow length of character variable

C	CALL SYMBEL(X,Y,DEGRES,HEIGHT,NCHAR,STRING)
C
C	THIS SUBROUTINE DRAWS A STRING OF CHARACTERS.
C
C	THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C	BELOW:
C        X,Y    - COORDINATES OF THE LOWER LEFT-HAND CORNER OF
C                 THE FIRST CHARACTER IN USERS UNITS.
C        DEGRES - ANGLE AT WHICH THE CHARACTERS ARE TO BE DRAWN
C                 IN DEGREES RELATIVE TO THE X-AXIS.
C        HEIGHT - HEIGHT OF THE CHARACTERS IN INCHES (OR MM).
C        NCHAR  - NUMBER OF CHARACTERS TO BE DRAWN.
C        STRING - ARRAY CONTAINING THE DISPLAY CODES OF THE
C                 CHARACTERS TO BE DRAWN.
C
	DIMENSION SX(12000),SY(12000),ASCII(127,4),BASE(4),RAST(4)
	INTEGER	  nsxsy
        PARAMETER ( nsxsy = 12000 )
	DIMENSION DTYPE(35),DLEN(35),IOFF(4),IFLEN(4),ITYPE(4)
C
	LOGICAL SYMROT,READIR,DOPLOT,COLRF
c	CHARACTER PLTFIL*80
	CHARACTER ICOM*2,ICHR*1,STRING*(*),DNME(35)*2,DNUM(35)*2
	CHARACTER DNAME(35)*80,KIF(4)*2,IFN(4)*2
	CHARACTER*1 KSUB,KSUPER,KBKSP,KCALL,PENS(16)

        REAL XBASE, YBASE
        INTEGER IPT
#ifdef unix
	INTEGER*2 SX,SY

        integer       dirlen
        character*255 fontdir
#else
	byte sx,sy
#endif
	INTEGER*2 ICNT,IBASE,IRAST
*added for linux port
	INTEGER*4 IBASE4,IRAST4
	INTEGER*2 ASCII
	INTEGER*2 DTYPE,DLEN,IOFF,IFLEN,ITYPE
	INTEGER COLOLD
	LOGICAL ASCFNT
	CHARACTER*265 err_msg
	INTEGER status

C
C
	INCLUDE 'PLTCOM.DAT'
        INCLUDE 'errmsg.parm'

C
	DATA DNUM,DNME/35*'@@',35*'@@'/
	DATA KSUB,KSUPER,KBKSP,KCALL/'_','^','\','@'/

	DATA KIF,IFN/'SR','SG','@@','@@','01','12','@@','@@'/
	DATA PENS/'1','2','3','4','5','6','7','8','9','A',
     *	'B','C','D','E','F','G'/
	DATA READIR/.FALSE./

* Mod to specify external references for SCALE and RANGE * Linux *kob* 1.97
        external scale, size

* Initialize 

        ICHR = ' '

C
C	DEFINE THE SCALE FACTOR TO BE THE AMOUNT OF MAGNIFICATION
C	REQUIRED TO CONVERT THE CHARACTERS IN THE CHARACTER DEFI-
C	NITION TABLE TO THE SIZE SPECIFIED BY THE USER.
C
        CALL INQSCL(XSCL,YSCL,ABIAS,BBIAS,XLO,YLO) ! Bug fix? 11.95
        CALL SCALE(XSCL,YSCL,ABIAS,BBIAS,0.,0.)

	COLRF=.FALSE.
C	IF(.NOT.READIR)GOTO 2020

* kob 1/97 - old 2020 code region now contained w/in if-then below.
	IF(.NOT.READIR) THEN
#ifdef unix
C *JD* Mod below fix for unix  -- use environment variable 3.19.91	
C *ACM* For backwards compatibility (LAS setup etc) also look for PLOTFONTS

	   call getenv ('FER_FONTS',fontdir)
       	   dirlen = lnblk (fontdir,255)
	   IF (dirlen.LE.1) THEN
       	      call getenv ('PLOTFONTS',fontdir)
       	      dirlen = lnblk (fontdir,255)
	   ENDIF

	   IF (dirlen .LE. 1) GOTO 5000

C *kob* 1/02 - cygwin 1.3.x requires "\" for unix directory accesses
#ifdef __CYGWIN__
	   OPEN(LUSY,FILE=fontdir(:dirlen)//'\fondat.dat',
     *	        STATUS='OLD')
#else
	   OPEN(LUSY,FILE=fontdir(:dirlen)//'/fondat.dat',
     *	        STATUS='OLD',ERR=5000)
#endif

#else
	   open(lusy,file='plotfonts:fondat.dat',readonly,
     *	       status='old',err=5000)
#endif
C
C	FILE EXISTS
C
	   READ(LUSY,983,END=1009)(DNUM(IJ),DNME(IJ),DTYPE(IJ),
     *	   DLEN(IJ),DNAME(IJ),IJ=1,35)
#ifdef unix
C Mod here from A to A9 for read of DNAME *JD* 3.8.91
C983	   FORMAT(1X,2A2,I2,I5,A)
983	   FORMAT(1X,2A2,I2,I5,A9)
#else
983	   format(1x,2a2,i2,i5,a30)
#endif
1009	   NFNT=IJ-1
    	   CLOSE(LUSY)
C
C	READ DEFAULT FONT
C
	   IOFF(1)=0
	   II=1
	   ID=1
	   READIR = .TRUE.			
	   call SETFONT(fontdir,dirlen,dname,ifont,ii,kif,dnme,
     .               ifn,dnum,itype,dtype,iflen, dlen,ioff,
     .               icnt,ascii,base,rast,iadd,
     .		     sx,sy,id)
	ENDIF

2000	SYMSC = HEIGHT * FACTOR / 21.0
C
C	ROTATE THE CHARACTER STRING.
C
	IF(DEGRES .NE. 0.0)GO TO 10
C
C	RESET THE CHARACTER ROTATION FLAG.
C
	SYMROT = .FALSE.
	GO TO 20
C
C	SET THE CHARACTER ROTATION FLAG.
C
10	SYMROT = .TRUE.
C
C	CONVERT THE ROTATION TO RADIANS.
C
	TEMP = 0.01745329251994 * DEGRES
C
C	COMPUTE THE SINE AND COSINE.
C
	SYMSIN = SIN(TEMP)
	SYMCOS = COS(TEMP)
C
C	SET VECTORS MODE.
C
20	PMODE = VECTOR
C
C	START AT THE LOWER LEFT-HAND CORNER OF THE FIRST CHARACTER.
C
	IF(NCHAR .EQ. 0) GOTO 300 	! Mod 11.30, was RETURN
	XTEMP = (X - XLO) * XSCALE	! was X * XSCALE
	YTEMP = (Y - YLO) * YSCALE	! was Y * YSCALE
	PENF = .FALSE.
	DOPLOT=NCHAR.GT.0
	IF(DOPLOT)CALL XYZPLT
C
C	INITIALIZE CONTROL VARIABLES.
C
	XORG = XTEMP
	YORG = YTEMP
	XREF = 0.0
	YREF = 0.0
	CHARSC = 1.0
	WIDTH = 0.0
	SLANT = 0.0
	IFONT = 1
	ASCFNT=ITYPE(1).EQ.-1
	SYMWID=0.0
	NCH=IABS(NCHAR)
C
C	DRAW THE CHARACTERS IN THE STRING ONE BY ONE.
C
	JJ=0
	DO 170 I=1,NCH
	JJ=JJ+1
	IF(JJ.GT.NCH)GOTO 180
	ICHR = STRING(JJ:JJ)
C
C	CHECK FOR A CONTROL CHARACTER.
C
	IF(ICHAR(ICHR) .EQ.27)THEN
	    ASCFNT=.FALSE.
	    GOTO 170
	ENDIF
	IF(ICHAR(ICHR) .EQ. 0)GOTO 180
#ifdef unix
	IF(.NOT.ASCFNT.AND.(ICHR.EQ.'@'.OR.
     *		(ICHR.GE.'\\'.AND.ICHR.LE.'_')))GOTO 80
#else
	IF(.NOT.ASCFNT.AND.(ICHR.EQ.'@'.OR.
     *		(ICHR.GE.'\'.AND.ICHR.LE.'_')))GOTO 80
#endif

* Some compliers store '\\' as a single character, others see it
* as '\'.  

	IF (KBKSP.NE.'\\') THEN
	   IF(.NOT.ASCFNT.AND.(ICHR.EQ.'@'.OR.
     *		(ICHR.GE.'\'.AND.ICHR.LE.'_')))GOTO 80
	ENDIF 
C
C	CALCULATE POINTER
C
	IF(ITYPE(IFONT).LE.0)THEN
C	    CHARACTER FONT
	    ICODE = ASCII(ICHAR(ICHR),IFONT)
	ELSE
C	    SYMBOL FONT
	    ICOM = STRING(JJ:JJ+1)
	    JJ=JJ+1
	    READ(ICOM,999,IOSTAT=ISTAT)II
999	    FORMAT(I2)
	    IF(ISTAT.EQ.0)THEN
	    	ICODE = ASCII(II,IFONT)
	    ELSE
	        ICODE = 1
	    ENDIF
	ENDIF
	IPOINT = ICODE+IOFF(IFONT)
	IPT = 0

	XMN=SX(IPOINT)
	XMX=SY(IPOINT)
	IF(XMN.GT.50.)XMN=XMN-100.
	IF(XMX.GT.50.)XMX=XMX-100.
	WIDTH=(XMX-XMN)*CHARSC*RAST(IFONT)
	SYMWID=SYMWID+WIDTH
	IF(.NOT.DOPLOT)GOTO 70
	IPOINT=IPOINT+1
	IF (IPOINT .GT. NSXSY) GOTO 70
	PENF = .FALSE.
C
C	DRAW THE CHARACTER
C
 30	CONTINUE
	IX=SX(IPOINT)
	IY=SY(IPOINT)
	IF(IX.GT.50)IX=IX-100
	IF(IY.GT.50)IY=IY-100
	IY=-IY
	IPOINT=IPOINT+1
	IF (IPOINT .GT. NSXSY) GOTO 70
        IPT = IPT + 1
C
C	CHECK FOR A CONTROL BYTE.
C
	IF(IX .EQ. 50)GO TO 60
C
C	CALCULATE THE NEW PEN COORDINATES RELATIVE TO THE LOWER
C	LEFT-HAND CORNER OF THE FIRST CHARACTER.
C
	XY=FLOAT(IY)-BASE(IFONT)
	XX=FLOAT(IX)-XMN+XY*SLANT
	XSYM =XREF+XX*CHARSC*RAST(IFONT)
	YSYM =YREF+XY*CHARSC*RAST(IFONT)
C
C	SCALE AND ROTATE THE NEW COORDINATES.
C
	IF(SYMROT)GO TO 40

	XTEMP = XSYM * SYMSC + XORG
	YTEMP = YSYM * SYMSC + YORG
	GO TO 50
40	XTEMP = (XSYM * SYMCOS - YSYM * SYMSIN) * SYMSC + XORG
	YTEMP = (XSYM * SYMSIN + YSYM * SYMCOS) * SYMSC + YORG
50	CONTINUE

* for decimal points, find the average x and Y location; will
* plot a point at this location.
C  xxx  
C  xxx  Dont do this: it causes trouble with encapsulated postscript.
C  xxx  and anyway, it was to correct trouble with the postscript translation
C  xxx  for one printer...
C  xxx
C  xxx        IF (ICHR .EQ. '.') THEN
C  xxx           IF (IPT .EQ. 1) THEN
C  xxx             XBASE = 0.
C  xxx             YBASE = 0.
C  xxx           ENDIF
C  xxx           XBASE = XBASE + XTEMP
C  xxx           YBASE = YBASE + YTEMP
C  xxx        ENDIF

C
C	MOVE THE PEN TO THE NEW COORDINATES.
C

        CALL XYZPLT

	PENF = .TRUE.
	GO TO 30
C
C	CHECK FOR THE END OF THE CHARACTER.
C
60	IF(IY .EQ. -50)GO TO 70
	PENF = .FALSE.
	GO TO 30

70	CONTINUE

C  End of character
C xxx If it was a decimal point, draw a tiny dot at its center.  
C  xxx
C  xxx        IF (ICHR .EQ. '.' .AND. IPT .GT. 1) THEN
C  xxx           XBASE = XBASE/ FLOAT(IPT-1)
C  xxx           YBASE = YBASE/ FLOAT(IPT-1)
C  xxx           XBASE = XBASE/ XSCALE
C  xxx           YBASE = YBASE/ YSCALE
C  xxx           call marker (1)              ! could call with 7 for bigger dot.
C  xxx           CALL PLOT(XBASE, YBASE, 0, 0)
C  xxx           CALL MARK(99)
C  xxx           call marker (1)
C  xxx        ENDIF
C
C	ADJUST THE REFERENCE POSITION TO TAKE INTO ACCOUNT THE
C	CHARACTER THAT WAS JUST DRAWN.
C

        XREF = XREF + WIDTH
	IF(CHARSC .EQ. 1.0)GO TO 169
C	XREF = XREF - 16.*(1.-CHARSC)*RAST(IFONT)
	YREF = 0.0
	CHARSC = 1.0
	GO TO 169
C
C	PROCESS CONTROL CHARACTERS.
C
80	IF(ICHR .NE. KSUB)GO TO 150
	CHARSC = 0.75
C	XREF = XREF + 16.*(1.-CHARSC)*RAST(IFONT)
	YREF=-11.*CHARSC*RAST(IFONT)
	GO TO 169
150	IF(ICHR .NE. KSUPER)GO TO 160
	CHARSC = 0.75
C	XREF = XREF + 16.*(1.-CHARSC)*RAST(IFONT)
	YREF=21.-11.*CHARSC*RAST(IFONT)
	GO TO 169
160	IF(ICHR .NE. KBKSP)GO TO 310
	SYMWID=SYMWID-WIDTH
	XREF = XREF - WIDTH
	WIDTH = 0.0
	GOTO 169
310	IF(ICHR.NE.KCALL)GOTO 169
C
C	UNPACK COMMAND
C
	ICOM = STRING(JJ+1:JJ+2)
	JJ=JJ+2
C
C	PROCESS COMMAND
C
	CALL UPPER(ICOM,2)
	IF(ICOM.EQ.'CL')THEN
C
C	    CLEAR ALL FONTS FROM THE BUFFERS
C
	    DO II=1,4
	    IOFF(II)=12000
	    IFLEN(II)=0
	    KIF(II)='@@'
	    IFN(II)='@@'
	    ITYPE(II)=0
	    BASE(II)=-9.
	    RAST(II)=1.
	    ENDDO

	    IOFF(1)=0
	    GOTO 169
        ELSE IF(ICOM(1:1).EQ.'C')THEN
C
C           LOOK FOR EXTENDED COLOR COMMAND OF THE FORM
C           @Cnnn WHERE nnn IS A 3 DIGIT NUMBER
C
            IF(.NOT.COLRF)COLOLD=COLOUR
            COLRF=.TRUE.
C
C           SEE IF THE FIRST CHARACTER IS A NUMBER
C
            DO 2050 K=1,9
                IF(ICOM(2:2).EQ.PENS(K))GOTO 2051
2050        CONTINUE
C
C           NO NUMBER FOUND - MUST BE A FONT CHANGE COMMAND
C
            IF(ICOM(2:2).NE.'0')GOTO 2010
2051        JJ=JJ+2
C
C           GET THE 3 DIGIT NUMBER
C
            READ(STRING(JJ-2:JJ),'(I3)',ERR=169)K
            IF(.NOT.DOPLOT)GOTO 169
            CALL COLOR(K)
	ELSE IF(ICOM(1:1).EQ.'P')THEN
C
C	    CHANGE PEN/COLOR
C
	    IF(.NOT.DOPLOT)GOTO 169
	    IF(.NOT.COLRF)COLOLD=COLOUR
	    COLRF=.TRUE.
	    DO 2030 K=1,16
	    IF(ICOM(2:2).EQ.PENS(K))GOTO 2031
2030	    CONTINUE
	    IF(ICOM(2:2).NE.'0')GOTO 2010
	    K=1
2031	    CALL COLOR(K)
	    GOTO 169
	ELSE IF(ICOM(1:1).EQ.'Z')THEN
	    DO 2040 K=1,16
	    IF(ICOM(2:2).EQ.PENS(K))GOTO 2041
2040	    CONTINUE
	    IF(ICOM(2:2).NE.'0')GOTO 2010
	    K=0
2041	    SLANT=SIN(K*0.08726646)
	    GOTO 169
	ENDIF
C	SEARCH FOR FONT BY NAME THEN BY NUMBER
2010	DO 1003 K=1,4
	IF(ICOM.EQ.KIF(K).OR.ICOM.EQ.IFN(K))GOTO 302
1003	CONTINUE
C	NOT FOUND SEARCH MAIN DIRECTORY
	DO 1004 ID=1,NFNT
	IF(ICOM.EQ.DNME(ID).OR.ICOM.EQ.DNUM(ID))GOTO 1001
1004	CONTINUE
C	FONT NOT FOUND USE LAST FONT
	GOTO 169
1001	ILEN=DLEN(ID)
C
C	FIND WHERE FONT WILL FIT
C
	DO 1005 II=4,1,-1
	IF(ILEN+IOFF(II).LE.12000) THEN
* *kob* 1/97 setfont call takes place of branch to old 1010
	    call SETFONT(fontdir,dirlen,dname,ifont,ii,kif,dnme,
     .               ifn,dnum,itype,dtype,iflen, dlen,ioff,
     .               icnt,ascii,base,rast,iadd,
     .		     sx,sy,id)
	    goto 169
	ENDIF
1005	CONTINUE
C
C	ERROR IF NO ROOM ANYWHERE!
C
	GOTO 169
c1006	IF(.NOT.NEWFIL)THEN
c	    INQUIRE(UNIT=LUHP,NAME=PLTFIL)
c	    CLOSE(LUHP)
c	ENDIF



c	IF(.NOT.NEWFIL)THEN
c	    IF(PTYPE.EQ.0 .OR. PTYPE.EQ.2 .OR. PTYPE.EQ.4)THEN
c		OPEN(LUHP,FILE=PLTFIL,FORM='UNFORMATTED',STATUS='OLD',
c     *		ACCESS='APPEND')
c	    ELSE
c		OPEN(LUHP,FILE=PLTFIL,STATUS='OLD',ACCESS='APPEND')
c	    ENDIF
c	ENDIF
C
C	DONE
C
C	IF(READIR)GOTO 169
C	READIR=.TRUE.
C	GOTO 2000
C
C	READ DIRECTORY
C

302	IFONT=K
169	ASCFNT=ITYPE(IFONT).EQ.-1
170	CONTINUE
180	IF(.NOT.DOPLOT)THEN
	    X=SYMWID*HEIGHT/21.
	    GOTO 300 ! Mod 11.30, was RETURN
	END IF
C
C	SCALE AND ROTATE THE FINAL COORDINATES.
C
	IF(SYMROT)GO TO 190
	XTEMP = XREF * SYMSC + XORG
	YTEMP = YORG
	GO TO 200
190	XTEMP = XREF * SYMCOS * SYMSC + XORG
	YTEMP = XREF * SYMSIN * SYMSC + YORG
C
C	PLOT TO THE END OF THE CHARACTER STRING.
C
200	PENF = .FALSE.
	CALL XYZPLT
C
C	RESET COLOR
C
	IF(COLRF.AND.(COLOLD.NE.COLOUR))CALL COLOR(COLOLD)
300     CALL SCALE(XSCL,YSCL,ABIAS,BBIAS,XLO,YLO) ! bug fix(?) 11.95

	RETURN
C
 5000   IF (dirlen .GT. 1) THEN
	   CALL WARN ('Could not open font file. Check the Ferret environment.')
	   err_msg = fontdir(:dirlen)//'/fondat.dat'
	ELSE
	   CALL WARN ('Fonts directory is undefined. Check the Ferret environment.')
	   err_msg = 'fondat.dat'
	ENDIF

        CALL errmsg (ferr_file_not_found, status, err_msg, *5002)

 5002	STOP

	END
