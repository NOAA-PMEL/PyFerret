	SUBROUTINE VIEW(X)
C**
C**    @(#)view.F	1.2    5/31/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/HD.INC'
	INCLUDE 'pplinc/CMRD.INC'
	INCLUDE 'pplinc/CMRDL.INC'
	INCLUDE 'pplinc/LABELS.INC'
	INCLUDE 'pplinc/LABCOM.INC'
	INCLUDE 'pplinc/HAB.INC'
	INCLUDE 'pplinc/PEN.INC'
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/VIEW.INC'
	INCLUDE 'cmdinc/SYSTEM.INC'
	INCLUDE 'cmdinc/SWITCH.INC'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:HD.INC'
	INCLUDE 'pplinc:CMRD.INC'
	INCLUDE 'pplinc:CMRDL.INC'
	INCLUDE 'pplinc:LABELS.INC'
	INCLUDE 'pplinc:LABCOM.INC'
	INCLUDE 'pplinc:HAB.INC'
	INCLUDE 'pplinc:PEN.INC'
	INCLUDE 'pplinc:PLT.INC'
	INCLUDE 'pplinc:VIEW.INC'
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:SWITCH.INC'
#endif
	IF(NX.EQ.0.OR.NY.EQ.0)RETURN
	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	IF(XYSCLE.EQ.1)THEN
	    XM=xtm
	    XMX=xtx
	ELSE
	    XM=ytm
	    XMX=ytx
	END IF
	IF(ZSCLE.NE.0)THEN
	    P1=1./ZSCLE
	ELSE
	    P1=(VEWZMX-VEWZMN)/(ytx-ytm)
	END IF
	XM=XM*P1
	XMX=XMX*P1
	YM=ytm*P1
	YMX=ytx*P1
C     TRANSFORM VIEWPOINT
	XV=(XMX-XM)*(VIEWX-xtm)/(xtx-xtm)+XM
	YV=(YMX-YM)*(VIEWY-ytm)/(ytx-ytm)+YM
	ICOUNT=ICOUNT+1
	CALL SURFCE(X,XM,XMX,NX,YM,YMX,NY,VEWZMN,VEWZMX,
     *	XV,YV,VIEWZ)
	RETURN
	END
