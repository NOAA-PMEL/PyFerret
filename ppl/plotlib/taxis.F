	SUBROUTINE TAXIS(IT,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *	XF,YHI,YF,TCSIZE)
C** 
C**    @(#)taxis.f	1.1    3/10/88
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

	CHARACTER*(*) IT1,IT2
	GOTO (10,20,20,30)IT+1
10	CALL TAXIS0(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,XF,YHI,YF,TCSIZE)
	RETURN
20	CALL TAXIS1(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,XF,YHI,YF,TCSIZE)
	RETURN
30	CALL TAXIS3(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,XF,YHI,YF,TCSIZE)
	RETURN
	END
