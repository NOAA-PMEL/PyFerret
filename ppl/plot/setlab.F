	SUBROUTINE SETLAB(XF,YF,LTT,LTB,LYL,LYR,LXT,LXB,IFX,IFY)
C** 
C**    @(#)setlab.F	1.1    3/10/88
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
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'pplinc/AXISL.INC'
#else
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:AXISL.INC'
#endif
	CHARACTER IFX*(*),IFY*(*)

	EXTERNAL scale


C     SETUP AXIS 
	LTT=0 
	LTB=0 
	LYL=0 
	LYR=0 
	LXT=0 
	LXB=0 
	IF(LABELY.EQ.-1)THEN 
	    LYL=-1 
	ELSE IF(LABELY.EQ.1)THEN 
	    LYR=1 
	END IF 
	IF(LABELX.EQ.-1)THEN 
	    LXB=-1 
	ELSE IF(LABELX.EQ.1)THEN 
	    LXT=1 
	END IF 
	IF(LABELT.LT.0)THEN 
	    LTB=LABELT 
	ELSE IF(LABELT.GT.0)THEN 
	    LTT=LABELT 
	END IF 
	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO) 
	IF(IFRX.EQ.' ')THEN 
	    CALL FRMT(XLO,XHI,NSIGX,IFX) 
	ELSE 
	    IFX=IFRX
	ENDIF 
	IF(IFRY.EQ.' ')THEN 
	    CALL FRMT(YLO,YHI,NSIGY,IFY) 
	ELSE 
	    IFY=IFRY
	ENDIF 
	RETURN
	END
