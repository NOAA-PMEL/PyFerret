	SUBROUTINE TAYIS(IT,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *	YF,XHI,XF,TCSIZE)
C** 
C**    @(#)tayis.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

* 	Mod 11.27.95 For 4 digit years J Davison
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER*(*) IT1,IT2
	GOTO (10,20,20,30)IT+1
10	CALL TAYIS0(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE)
	RETURN
20	CALL TAYIS1(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE)
	RETURN
30	CALL TAYIS3(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE)
	RETURN
	END
