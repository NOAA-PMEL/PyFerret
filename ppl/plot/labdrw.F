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
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/LABELS.INC'
	INCLUDE 'pplinc/LABCOM.INC'
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/PLTL.INC'
	INCLUDE 'pplinc/HAB.INC'
	INCLUDE 'pplinc/LINES.INC'
	INCLUDE 'pplinc/MPLOT.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'pplinc/TICS.INC'
	INCLUDE 'pplinc/TICS2.INC'
	INCLUDE 'pplinc/AXISL.INC'
	INCLUDE 'pplinc/AXLWID.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:LABELS.INC'
        INCLUDE 'pplinc:LABCOM.INC'
        INCLUDE 'pplinc:PLT.INC'
        INCLUDE 'pplinc:PLTL.INC'
        INCLUDE 'pplinc:HAB.INC'
        INCLUDE 'pplinc:LINES.INC'
        INCLUDE 'pplinc:MPLOT.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:TICS.INC'
        INCLUDE 'pplinc:TICS2.INC'
        INCLUDE 'pplinc:AXISL.INC'
        INCLUDE 'pplinc:AXLWID.INC'
#endif
	CHARACTER IFX*(*),IFY*(*)
C	DRAW LABELS
	IF(MAKEP.EQ.'VIEW')GOTO 20
	IF(LABELY.EQ.0.OR.ITFLG.EQ.-1)GOTO 10
	NCHR=LNBLK(YLAB,255)
	IF(LABELY.LT.0.)THEN
	    XL=XLO-(YWID+HYLAB)/XF
	ELSE
	    XL=XHI+(YWID+2.*HYLAB)/XF
	ENDIF
	YL=YLO+(YLEN-SYMWID(HYLAB,NCHR,YLAB))*.5/YF
	CALL SYMBEL(XL,YL,90.,HYLAB,NCHR,YLAB)
10	IF(LABELX.EQ.0.OR.ITFLG.EQ.1)GOTO 20
	NCHR=LNBLK(XLAB,255)
	IF(LABELX.LT.0)THEN
	    YL=YLO-(XWID+2.*HXLAB)/YF
	ELSE
	    YL=YHI+(XWID+HXLAB)/YF
	ENDIF
	XL=XLO+(XLEN-SYMWID(HXLAB,NCHR,XLAB))*.5/XF
	CALL SYMBEL(XL,YL,0.,HXLAB,NCHR,XLAB)
20	NCHR=LNBLK(LAB1,255)
	XL=XLO+(XLEN-SYMWID(HLAB1,NCHR,LAB1))*.5/XF
C
C	DETERMINE POSITION OF MAIN LABEL
C
	ITFLG2=0
	IF(ITFLG.EQ.1)ITFLG2=1
	IF (LABELX .GE. 0) THEN
	  YTEMP = XWID - 1.9*ABS(XCSIZE) + 2.5*HLAB1
	ELSE
	  YTEMP = XWID + 2.0*HXLAB * (1-ITFLG2)
     *	          + 2.5*HLAB1
	ENDIF
	YL=YLO-YTEMP/YF
	CALL SYMBEL(XL,YL,0.,HLAB1,NCHR,LAB1)
70	RETURN
	END
