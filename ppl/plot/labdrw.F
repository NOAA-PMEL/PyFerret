	SUBROUTINE LABDRW(XF,YF,IFX,IFY)
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
C**    @(#)labdrw.F	1.1    3/10/88
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
C *acm* v552 3/03 Longer strings and changes for multi-line labels.

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'AXIS.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'PLTL.INC'
	INCLUDE 'HAB.INC'
	INCLUDE 'LINES.INC'
	INCLUDE 'MPLOT.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'TICS.INC'
	INCLUDE 'TICS2.INC'
	INCLUDE 'AXISL.INC'
	INCLUDE 'AXLWID.INC'

	CHARACTER IFX*(*),IFY*(*)
        INTEGER j, lablen, numlines, slen
        REAL htnew, wid

c        INTEGER lnbeg(500), lnend(500)
c        CHARACTER fontprefix*3, penprefix*3, newfont*3, newpen*3
c        REAL line_spacing

c        line_spacing = 0.4

        lablen = 2048

C	DRAW LABELS
	IF(MAKEP.EQ.'VIEW')GOTO 40
	IF(LABELY.EQ.0.OR.ITFLG.EQ.-1)GOTO 20

C  Y label. center each line 

        CALL tm_break_lines (ylab, lnbeg, lnend, numlines)

        IF(LABELY.LT.0.)THEN 
        XL=XLO-(YWID+HYLAB)/XF
           xl = xl - (numlines-1)*hylab/xf
        ELSE
           XL=XHI+(YWID+2.*HYLAB)/XF
        ENDIF

        nfont = 0
        npen = 0
        fontprefix = ""
        penprefix = ""
        DO 10 j = 1, numlines
           CALL getlabfonts (ylab, lnbeg, lnend, j, npnew, nfnew, 
     .           newfont, newpen, nchr)

	   YL=YLO+(YLEN-SYMWID(HYLAB, NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 ylab(lnbeg(j):lnend(j)) ))*.5/YF

	   CALL SYMBEL(XL, YL, 90., HYLAB, NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 ylab(lnbeg(j):lnend(j)))

           XL= xl + line_spacing*HYLAB/XF

C  Set up to use any font settings from this line on subsequent lines.

           IF (npnew.GT.0) THEN
              penprefix(1:3) = newpen(1:3)
              npen = 3
           ENDIF
           IF (nfnew.GT.0) THEN
              fontprefix(1:3) = newfont(1:3)
              nfont = 3
           ENDIF

10      CONTINUE


C  X label

20	IF(LABELX.EQ.0.OR.ITFLG.EQ.1)GOTO 40
        CALL tm_break_lines (xlab, lnbeg, lnend, numlines)

	IF(LABELX.LT.0)THEN
	    YL=YLO-(XWID+2.*HXLAB)/YF
	ELSE
	    YL=YHI+(XWID+HXLAB)/YF
            IF (numlines .GT. 1) yl = yl + 
     .           line_spacing*(numlines-1)* hxlab/yf
	ENDIF

        nfont = 0
        npen = 0
        fontprefix = ""
        penprefix = ""

        DO 30 j = 1, numlines
	IF(LABELX.LT.0)THEN

           CALL getlabfonts (xlab, lnbeg, lnend, j, npnew, nfnew, 
     .           newfont, newpen, nchr)

	   XL=XLO+(XLEN-SYMWID(HXLAB,NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 xlab(lnbeg(j):lnend(j)) ))*.5/XF

	   CALL SYMBEL(XL,YL,0.,HXLAB, NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 xlab(lnbeg(j):lnend(j)))

           yl = yl - line_spacing* hxlab/yf

C  Set up to use any font settings from this line on subsequent lines.

           IF (npnew.GT.0) THEN
              penprefix(1:3) = newpen(1:3)
              npen = 3
           ENDIF
           IF (nfnew.GT.0) THEN
              fontprefix(1:3) = newfont(1:3)
              nfont = 3
           ENDIF

        ENDIF
30      CONTINUE

C  Main label

40      CALL tm_break_lines (lab1, lnbeg, lnend, numlines)

* Adjust character size for the length of the longest line
* This may have already been done in box_label, but not if
* title was set by a PPL TITLE command.

        htnew = hlab1
        wid = 0.
        DO 330 k = 1, numlines
           slen = lnend(k) - lnbeg(k) + 1
	   wid = MAX(wid, SYMWID(htnew, slen, lab1(lnbeg(k):lnend(k))))
  330   CONTINUE

* xsize in call to box_label is .75* x axis length

        htnew = htnew * MIN(1., 1.25* .75*xlen/wid)

C	Determine starting y position of main label

        ITFLG2=0
        IF(ITFLG.EQ.1)ITFLG2=1
	IF (LABELX .GE. 0) THEN
	  YTEMP = XWID - 1.9*ABS(XCSIZE) + 2.5*htnew
	ELSE
	  YTEMP = XWID + 2.0*HXLAB * (1-ITFLG2)
     .	          + 2.5*htnew
	ENDIF
	YL=YLO-YTEMP/YF

        nfont = 0
        npen = 0
        fontprefix = ""
        penprefix = ""

        DO 50 j = 1, numlines
           CALL getlabfonts (lab1, lnbeg, lnend, j, npnew, nfnew, 
     .           newfont, newpen, nchr)
	   XL=XLO+(XLEN-
     .          SYMWID(htnew,NCHR,lab1(lnbeg(j):lnend(j))))*.5/XF

	   CALL SYMBEL(XL,YL,0.,htnew, NCHR+npen+nfont,
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 lab1(lnbeg(j):lnend(j)))

           yl = yl - line_spacing* htnew/yf

C  Set up to use any font settings from this line on subsequent lines.

           IF (npnew.GT.0) THEN
              penprefix(1:3) = newpen(1:3)
              npen = 3
           ENDIF
           IF (nfnew.GT.0) THEN
              fontprefix(1:3) = newfont(1:3)
              nfont = 3
           ENDIF
50      CONTINUE

70	RETURN
	END
