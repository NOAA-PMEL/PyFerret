	SUBROUTINE PRMTIV(X,NXS)
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
C**    @(#)prmtiv.F	1.1    3/10/88

* modified 6/99 *sh*
*	added mouse input (CALL IGRNPT) to %LABEL
*	added %WHERE mouse input command
* v552 3/03 *acm* changes for multi-line labels with %label, %xaxis, %yaxis
*                   and remove VMS include statements.
* v554 3/04 *acm* 5th arg of %xaxis, %yaxis should set the # of tics
* v554 3/04 *acm* Need to send parameters used for making graticule lines
*                 to xaxis, yaxis, initialized to values for graticule off.
* v580 10/05 *acm* Fix for bug1003; pen may be changed by @Cnnn as well
*                  as @Pn. In saving info about color/font changes from
*                  one line to the next in multi-line lables, allow for
*                  @Cnnn syntax.  (changes also in getlabfonts)
* v580 11/05 *acm* Send new arg to xaxis1,yaxis1 DMS (arg is 0 here)
C        1/06  xvaloff,yvaloff add this amount to each x or y axis label. 
C              (axis data was read as double precision and converted to single 
C              after applying an offset.)
* V683 12/12 *acm* Fix for applying a font setting in multiple line labels
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V69+  5/14 *acm* Fix for ascii font in multi-line labels, ticket 2176
C V695 *acm*  5/15 Change in arguments to YAXIS1 and XAXIS1
* v695 9/15 *acm* write values to PPL* symbols using 6 digits not 4
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.
* 12/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals. In PyFerret pltype is 3

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
C	This subroutine does plot primitives.  They
C	should be used with care since interactions are
C	a very strong possibility.
C
C	These routines do not work with multplt!!!!
C
c
c	%OPNPLT
c		/overlay	do not erase last plot
c
c	%CLSPLT
c		/wait		immediatly return to get next command
c
c	%RANGE
c		min		minimum value of data to be ranged
c		max		maximum value of data
c		ntic		number of large increments
c	ppl$range_low		new minimum range value
c	ppl$range_high		new maximum range value
c	ppl$range_inc		new increment
c
c	%PLTLIN
c		n		plot line n using current scales
c
c	%LABEL
c		/user		x and y in user or inches
c		x		x position user or inches
c		y		y position user or inches
c		ipos		-1 left, 0 center, +1 right justify
c		ang		angle to draw label degrees
c		chsize		character size inches
c		label		string to draw
c
c	%XAXIS
c		/user		y in user or inches
c		xlow		min value of x user
c		xhigh		max value of x user
c		xtic		large tic increment user
c		y		y position user or inches
c		nmstc		number of small tics
c		lint		label interval (large tics)
c		xunit		divisor for axis label
c		ipos		-1 bottom, 0 none, +1 top of label
c		csize		character size inches
c		frmt		axis format char*20
c
c	%YAXIS
c		/user		x in user or inches
c		ylow		min value of y user
c		yhigh		max value of y user
c		ytic		large tic increment user
c		x		x position user or inches
c		nmstc		number of small tics
c		lint		label interval (large tics)
c		yunit		divisor for axis label
c		ipos		-1 left, 0 none, +1 right of label
c		csize		character size inches
c		frmt		axis format char*20
c
c	%WHERE
c		N/A

	DIMENSION X(NXS)
C
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'LINES.INC'
	INCLUDE 'PEN.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'PLTL.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'HD.INC'
	INCLUDE 'MISS.INC'
	INCLUDE 'TAXIS.INC'
	include 'axis_inc.decl'
	INCLUDE 'AXIS.INC'
	INCLUDE 'AXISL.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'CMRDL.INC'
	INCLUDE 'PRMCOM.INC'
	INCLUDE 'TICS.INC'
	INCLUDE 'AXLWID.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'SWITCH.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'gkscm1_inc.decl'  ! with wsid for igrnpt
	INCLUDE 'GKSCM1.INC'
	INCLUDE 'pyfonts.cmn'

C
C
	CHARACTER FRMTX*20,SYM*120
	LOGICAL SVQIET,SVLOG,CLEAR

	LOGICAL mouse_input, need_justify, need_arrow
	INTEGER	IGRNPT, idummy, ichoice, iat
	REAL xp, yp, across
        INTEGER numlines, i, j, dms
        LOGICAL lineflag

	EXTERNAL range, scale, size

c        line_spacing = 1.4

C***	OPNPLT
C
	IF(ICOM.EQ.'%OPNPLT')THEN
	    SVQIET=QUIETF
	    SVLOG=LOGCMF
	    QUIETF=.TRUE.
	    LOGCMF=.FALSE.
	    CLEAR = .NOT.SOVER
	    PLTOPN=.TRUE.
	    IF(.NOT.SOVER)CALL SIZE(WIDTH,HEIGHT)
	    CALL OPNDEV(TERMF,CLEAR)
	    CALL COLOR(IPEN(0))
C
C***	CLSPLT
C
	ELSE IF(ICOM.EQ.'%CLSPLT')THEN
	    QUIETF=SVQIET
	    LOGCMF=SVLOG
	    PLTOPN=.FALSE.
	    CALL HABIS
C
C***	RANGE
C
	ELSE IF(ICOM.EQ.'%RANGE')THEN
	    RMIN=P(1)
	    RMAX=P(2)
	    NTIC=P(3)
	    if(rmin.eq.rmax)then
		rlow=rmin
		rhigh=rmax
	    else
	        CALL RANGE(RMIN,RMAX,NTIC,RLOW,RHIGH,RINC)
	    endif
	    SYM='*PPL$RANGE_LOW'
	    CALL PUTVAL(SYM,RLOW,7,IER)
C
	    SYM='*PPL$RANGE_HIGH'
	    CALL PUTVAL(SYM,RHIGH,7,IER)
C
	    SYM='*PPL$RANGE_INC'
	    CALL PUTVAL(SYM,RINC,7,IER)
C
C***	XAXIS
C
	ELSE IF(ICOM.EQ.'%XAXIS')THEN

            dms = 0
            DO 100 ii = 1, 4
              hgrat(ii,1) = 0.
              hgrat(ii,2) = 0.
              vgrat(ii,1) = 0.
              vgrat(ii,2) = 0.
100         CONTINUE

            hcolor(1) = 1 
            hcolor(2) = 1
            vcolor(1) = 1
            vcolor(2) = 1
            across = 0.
            lineflag = .FALSE.

	    CALL INQSCL(XFF,YFF,XOR,YOR,XLOW,YLOW)
	    IF(M.GE.1.AND.IFLG(1).NE.0)THEN
		XLOW=P(1)
	    ELSE
		XLOW=XLO
	    ENDIF
	    IF(M.GE.2.AND.IFLG(2).NE.0)THEN
		XHIGH=P(2)
	    ELSE
		XHIGH=XHI
	    ENDIF
	    IF(M.GE.3.AND.IFLG(3).NE.0)THEN
		XTICC=P(3)
	    ELSE
		XTICC=XTIC
	    ENDIF
	    IF(M.GE.4.AND.IFLG(4).NE.0)THEN
		IF(SUSER)THEN
		    Y=P(4)
		ELSE
		    Y=YLOW+P(4)/YFF
		ENDIF
	    ELSE
		Y=YLOW
	    ENDIF
	    IF(M.GE.5.AND.IFLG(5).NE.0)THEN
		NSMTCS=P(5)
	    ELSE
		NSMTCS=NMTCX
	    ENDIF
	    IF(M.GE.6.AND.IFLG(6).NE.0)THEN
		LINTT=P(6)
	    ELSE
		LINTT=LINTX
	    ENDIF
	    IF(M.GE.7.AND.IFLG(7).NE.0)THEN
		XUNIT=P(7)
	    ELSE
		XUNIT=1.
	    ENDIF
	    IF(M.GE.8.AND.IFLG(8).NE.0)THEN
		IPOS=P(8)
	    ELSE
		IPOS=LABELX
	    ENDIF
	    IX=IPOS
	    IF(M.GE.9.AND.IFLG(9).NE.0)THEN
		CHSIZE=P(9)
	    ELSE
		CHSIZE=XCSIZE
	    ENDIF
	    IF(N.GT.0)THEN
		FRMTX=LABEL(1:N)
	    ELSE IF(IFRX.NE.' ')THEN
		FRMTX=IFRX
	    ELSE
		CALL FRMT(XLOW,XHIGH,NSIGX,FRMTX)
	    ENDIF
	    XFF=XLEN/(XHIGH-XLOW)
	    CALL SCALE(XFF,YFF,XOR,YOR,XLOW,YLOW)
	    call color(ipen(0))
	    CALL XAXIS1(XLOW,XHIGH,XTICC,NSMTCS,LINTT,XUNIT,
     *		IPOS,FRMTX,XFF,Y,YFF,CHSIZE,across,xdms,xdspac,vgrat,
     *		vcolor,vline,xvaloff,ipen(0))

            CALL tm_break_lines (xlab, lnbeg, lnend, numlines)
            nfont = 0
            npen = 0
            fontprefix = ""
            penprefix = ""
            DO 20 j = 1, numlines
               CALL getlabfonts (xlab, lnbeg, lnend, j, fontprefix, 
     .           npnew, nfnew, newfont, newpen, nchr)

	       IF(IPOS.LT.0)THEN
		  YL=Y-(XWID+2.0*HXLAB)/YFF
	       ELSE
	 	  YL=Y+(XWID+HXLAB)/YFF
	       ENDIF
               YL = YL - line_spacing*(j-1)*HYLAB/yff
	       XL=XLOW+(XLEN-SYMWID(HXLAB, NCHR+npen+nfont,
     .                  penprefix(1:npen)//fontprefix(1:nfont)//
     .                  xlab(lnbeg(j):lnend(j) )) )*.5/XFF

               CALL SYMBEL(XL,YL,0.,HXLAB,NCHR+npen+nfont,
     .                  penprefix(1:npen)//fontprefix(1:nfont)//
     .                  xlab(lnbeg(j):lnend(j)))

C  Set up to use any font settings from this line on subsequent lines.

               IF (npnew.GT.0) THEN
                  penprefix(1:npnew) = newpen(1:npnew)
                  npen = npnew
               ENDIF
               IF (nfnew.GT.0) THEN
                  fontprefix(1:nfnew) = newfont(1:nfnew)
                  nfont = nfnew
               ENDIF

  20        CONTINUE

C
C***	YAXIS
C
	ELSE IF(ICOM.EQ.'%YAXIS')THEN
        
            dms = 0
            DO 200 ii = 1, 4
              hgrat(ii,1) = 0.
              hgrat(ii,2) = 0.
              vgrat(ii,1) = 0.
              vgrat(ii,2) = 0.
200         CONTINUE

            hcolor(1) = 1 
            hcolor(2) = 1
            vcolor(1) = 1
            vcolor(2) = 1
            across = 0.
            lineflag = .FALSE.

	    CALL INQSCL(XFF,YFF,XOR,YOR,XLOW,YLOW)
	    IF(M.GE.1.AND.IFLG(1).NE.0)THEN
		YLOW=P(1)
	    ELSE
		YLOW=YLO
	    ENDIF
	    IF(M.GE.2.AND.IFLG(2).NE.0)THEN
		YHIGH=P(2)
	    ELSE
		YHIGH=YHI
	    ENDIF
	    IF(M.GE.3.AND.IFLG(3).NE.0)THEN
		YTICC=P(3)
	    ELSE
		YTICC=YTIC
	    ENDIF
	    IF(M.GE.4.AND.IFLG(4).NE.0)THEN
		IF(SUSER)THEN
		    XX=P(4)
		ELSE
		    XX=XLOW+P(4)/XFF
		ENDIF
	    ELSE
		XX=XLOW
	    ENDIF
	    IF(M.GE.5.AND.IFLG(5).NE.0)THEN
		NSMTCS=P(5)
	    ELSE
		NSMTCS=NMTCY
	    ENDIF
	    IF(M.GE.6.AND.IFLG(6).NE.0)THEN
		LINTT=P(6)
	    ELSE
		LINTT=LINTY
	    ENDIF
	    IF(M.GE.7.AND.IFLG(7).NE.0)THEN
		YUNIT=P(7)
	    ELSE
		YUNIT=1.
	    ENDIF
	    IF(M.GE.8.AND.IFLG(8).NE.0)THEN
		IPOS=P(8)
	    ELSE
		IPOS=LABELY
	    ENDIF
	    IY=IPOS
	    IF(M.GE.9.AND.IFLG(9).NE.0)THEN
		CHSIZE=P(9)
	    ELSE
		CHSIZE=YCSIZE
	    ENDIF

c	    IF(N.GT.0)THEN
c		FRMTX=LABEL(1:N)
c	    ELSE IF(IFRX.NE.' ')THEN  <-- bug here: check IFRY instead.
c		FRMTX=IFRY
c	    ELSE
c		CALL FRMT(YLOW,YHIGH,NSIGY,FRMTX)
c	    ENDIF


	    IF(N.GT.0)THEN
		FRMTX=LABEL(1:N)
	    ELSE IF(IFRY.NE.' ')THEN
		FRMTX=IFRY
	    ELSE
		CALL FRMT(YLOW,YHIGH,NSIGY,FRMTX)
	    ENDIF

	    YFF=YLEN/(YHIGH-YLOW)
	    CALL SCALE(XFF,YFF,XOR,YOR,XLOW,YLOW)
	    call color(ipen(0))
	    CALL YAXIS1(XX,XFF,YLOW,YHIGH,YTICC,NSMTCS,LINTT,YUNIT,
     *		IPOS,FRMTX,YFF,CHSIZE,across,ydms,ydspac,hgrat,hcolor,hline,
     *          yvaloff,.FALSE.,SHRINKY,ipen(0))

            CALL tm_break_lines (ylab, lnbeg, lnend, numlines)
            nfont = 0
            npen = 0
            fontprefix = ""
            penprefix = ""
            DO 30 j = 1, numlines
               CALL getlabfonts (ylab, lnbeg, lnend, j, fontprefix, 
     .           npnew, nfnew, newfont, newpen, nchr)

	       IF(IPOS.LT.0)THEN
	           XL=XX-(YWID+HYLAB)/XFF
	       ELSE
	           XL=XX+(YWID+2.0*HYLAB)/XFF
	       ENDIF 
               XL = XL + line_spacing*(j-1)*hylab/xff
	       YL=YLOW+(YLEN-SYMWID(HYLAB,NCHR+npen+nfont,
     .                  penprefix(1:npen)//fontprefix(1:nfont)//
     .                  ylab(lnbeg(j):lnend(j) )) )*.5/XFF

               CALL SYMBEL(XL,YL,90.,HYLAB,NCHR+npen+nfont,
     .                  penprefix(1:npen)//fontprefix(1:nfont)//
     .                  ylab(lnbeg(j):lnend(j)))

C  Set up to use any font settings from this line on subsequent lines.

               IF (npnew.GT.0) THEN
                  penprefix(1:npnew) = newpen(1:npnew)
                  npen = npnew
               ENDIF
               IF (nfnew.GT.0) THEN
                  fontprefix(1:nfnew) = newfont(1:nfnew)
                  nfont = nfnew
               ENDIF

  30        CONTINUE

C
C***	PLTLIN
C
	ELSE IF(ICOM.EQ.'%PLTLIN')THEN
	    IF(M.GE.1.AND.IFLG(1).NE.0)THEN
		I=P(1)
	    ELSE
		I=1
	    ENDIF
	    IF(IWIND.GT.0)THEN
	    	CALL INQSCL(XFF,YFF,XOR,YOR,XLOW,YLOW)
		XHIGH=XLOW+XLEN/XFF
		YHIGH=YLOW+YLEN/YFF
		CALL WINDOW(XLOW,YLOW,XHIGH,YHIGH)
	    ENDIF
	IBASE=0
	IY=NXS/2
C
	DO 50 J=1,I-1
	  IBASE=IBASE+LLENG(J)
50	CONTINUE
C
	  CALL MARKH(HMARK(I))
	  CALL COLOR(IPEN(I))
C
C     SET PLOT MODE
	    IF(ITYPE(I).GE.4)THEN
	      CALL DASHSZ(DN1(I),UP1(I),DN2(I),UP2(I))
	      CALL DASHES
	    ELSE
	      CALL VECTRS
	    END IF
	    IF(ITYPE(I).EQ.3)CALL POINTS
	    IF(.NOT.(XLE.OR.XEQ.OR.XGE.OR.YLE.OR.YEQ.OR.YGE))THEN
C     DO NOT CHECK POINTS FOR MISSING DATA
	      CALL TRANS(I,X(IBASE+1),X(IY+IBASE+1),XT,YT)
	      CALL PLOT(XT,YT,0,0)
	      IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
	      IM=0
	      IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	      DO 60 J=IBASE+1,IBASE+LLENG(I)
	        CALL TRANS(I,X(J),X(IY+J),XT,YT)
	        CALL PLOT(XT,YT,1,0)
	        CALL MARK(IM)
60	      CONTINUE
	    ELSE
C     CHECK POINTS FOR OUT OF RANGE   (CM)
	      DO 70 J=IBASE+1,IBASE+LLENG(I)
		IF(XLE.AND.X(J).LE.CMXLE)GOTO 70
		IF(XEQ.AND.X(J).EQ.CMXEQ)GOTO 70
		IF(XGE.AND.X(J).GE.CMXGE)GOTO 70
		IF(YLE.AND.X(IY+J).LE.CMYLE)GOTO 70
		IF(YEQ.AND.X(IY+J).EQ.CMYEQ)GOTO 70
		IF(YGE.AND.X(IY+J).GE.CMYGE)GOTO 70
	        CALL TRANS(I,X(J),X(IY+J),XT,YT)
	        CALL PLOT(XT,YT,0,0)
	        IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
 	        IM=0
	        IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	        GOTO 71
70	      CONTINUE
71	      IF=1
	      DO 72 K=J+1,IBASE+LLENG(I)
		IF((XLE.AND.X(K).LE.CMXLE).OR.
     *		   (XEQ.AND.X(K).EQ.CMXEQ).OR.
     *		   (XGE.AND.X(K).GE.CMXGE).OR.
     *		 (YLE.AND.X(IY+K).LE.CMYLE).OR.
     *		 (YEQ.AND.X(IY+K).EQ.CMYEQ).OR.
     *		 (YGE.AND.X(IY+K).GE.CMYGE))THEN
	          IF=0
	        ELSE
	          CALL TRANS(I,X(K),X(IY+K),XT,YT)
	          CALL PLOT(XT,YT,IF,0)
	          CALL MARK(IM)
	          IF(IF.EQ.0)THEN
	            IF=1
	          END IF
	        END IF
72	      CONTINUE
	    END IF
	    IF(ITYPE(I).EQ.2.OR.ITYPE(I).EQ.5)CALL MARK(IMARK(I))
 	CALL COLOR(IPEN(0))
	CALL WINDOW(0.,0.,0.,0.)
C
C***	%LABEL
C
	ELSE IF(ICOM.EQ.'%LABEL')THEN
	    IF(N.EQ.0)RETURN

* (PyFerret) moveable labels style 
	    IF (pyfont .AND. pyf_set(imove)) igroup = imove

* get missing inputs interactively -- through mouse input (6/99)
* ... if the user **explicitly** omits the justification (iflg(3)) then
*	put up menu (including arrow options)
	    mouse_input = m.LE.2 .OR. iflg(1).EQ.0 .OR. iflg(2).EQ.0
	    need_arrow = .FALSE.
	    IF ( mouse_input ) THEN
	      need_justify = m.GE.3 .AND. iflg(3).EQ.0
	      ichoice = IGRNPT(wsid, need_justify, xp, yp)	! mouse input
	      need_arrow =   ichoice.EQ.70 .OR. ichoice.EQ.102
     .			.OR. ichoice.EQ.76 .OR. ichoice.EQ.108
	      IF (need_arrow) THEN
* ... user wants a line pointing to the text
	         idummy = IGRNPT(wsid, .FALSE., p(1), p(2))
	         p(3) = -1	! left justify
	         IF( ichoice.EQ.70 .OR. ichoice.EQ.102 )THEN	 ! arow head?
	           iat = 2
	         ELSE
	           iat = 1
	         ENDIF
	      ELSE
* ... user just wants text
	         p(1) = xp
	         p(2) = yp
	         IF (ichoice .EQ. 67 .OR. ichoice.EQ.99) THEN
	           p(3) = 0	! center
	         ELSE  IF (ichoice .EQ. 82 .OR. ichoice.EQ.114) THEN
	           p(3) = 1	! right justify
	         ELSE	
	           p(3) = -1	! left justify
	         ENDIF
	      ENDIF
	      iflg(1) = 1
	      iflg(2) = 1
	      iflg(3) = 1
	      IF (m.LT.3) m = 3
	    ENDIF

C
C	SET SCALE FOR LABEL
C
	    CALL INQSCL(XFF,YFF,XOR,YOR,XLOW,YLOW)
	    IF(M.GE.3.AND.IFLG(3).NE.0)THEN
		IPOS=P(3)
	    ELSE
	    	IPOS=-1
	    ENDIF
	    IF(M.GE.4.AND.IFLG(4).NE.0)THEN
	    	ANG=P(4)
	    ELSE
	        ANG=0.
	    ENDIF
	    IF(M.GE.5.AND.IFLG(5).NE.0)THEN
	    	CHSIZE=P(5)
	    ELSE
	    	CHSIZE=HLAB1
	    ENDIF
	    IF(SUSER .OR. mouse_input)THEN	! mouse is always /USER
	    	XFUSER=1.
		YFUSER=1.
		YLOUSR=0.
		XLOUSR=0.
	    ELSE
	    	XFUSER=XFF
		YFUSER=YFF
		XLOUSR=XLOW
		YLOUSR=YLOW
	    ENDIF
	    CS=COS(ANG*0.0175329252)
	    SN=SIN(ANG*0.0175329252)

            CALL tm_break_lines (label, lnbeg, lnend, numlines)
            nfont = 0
            npen = 0
            fontprefix = ""
            penprefix = ""
            DO 40 j = 1, numlines
               CALL getlabfonts (label, lnbeg, lnend, j, fontprefix, 
     .           npnew, nfnew, newfont, newpen, nchr)
               XMN=SYMWID(CHSIZE,NCHR+npen+nfont, 
     .           penprefix(1:npen)//fontprefix(1:nfont)// 
     .           label(lnbeg(j):lnend(j))) 

	       SHFT=0.0
	       IF(IPOS.GT.0)THEN
	   	   SHFT=-XMN
	       ELSE IF(IPOS.EQ.0)THEN
		   SHFT=-XMN*.5
	       ENDIF

	       XT=P(1)/XFUSER+XLOUSR+CS*SHFT/XFF + 
     .             line_spacing*(j-1)*chsize*SN/xff
	       YT=P(2)/YFUSER+YLOUSR+SN*SHFT/YFF - 
     .             line_spacing*(j-1)*chsize*CS/yff

	       IF (need_arrow .AND. j.EQ.1) CALL ARROW(p(1),p(2),xp,yp,
     .				IAT,SHFT,XMN,CS,SN,XFF,YFF,chsize)

               CALL SYMBEL(XT,YT,ANG,CHSIZE,NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 label(lnbeg(j):lnend(j)))

C  Set up to use any font settings from this line on subsequent lines.

               IF (npnew.GT.0) THEN
                  penprefix(1:npnew) = newpen(1:npnew)
                  npen = npnew
               ENDIF
               IF (nfnew.GT.0) THEN
                  fontprefix(1:nfnew) = newfont(1:nfnew)
                  nfont = nfnew
               ENDIF

  40        CONTINUE
  	    
* (PyFerret) cancel tic-mark text style
	   IF (pyfont) igroup = fnt_all

C
C
C***	%WHERE
C
	ELSE IF(ICOM.EQ.'%WHERE')THEN
	    idummy = IGRNPT(wsid, .FALSE., p(1), p(2))	! mouse input

	ENDIF

	RETURN
	END
c!!!!!!!!!!!!!!!!!!!!!
