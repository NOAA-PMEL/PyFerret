	SUBROUTINE LABDRW(XF,YF,IFX,IFY)
C** 
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
