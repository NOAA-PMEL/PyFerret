      FUNCTION NUMDM1(IT) 
C** 
C**    @(#)numdm1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

* 	Mod J Davison 11.22.95 for 4 digit years
* 	Mod J Davison  2.6.96  fix bug re 400 year biz
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

C     THIS FUNCTION RETURNS NO. OF DAYS IN CURRENT MONTH
 	CHARACTER*(*) IT
      DIMENSION ID(12) 
      DATA ID/31,28,31,30,31,30,31,31,30,31,30,31/ 
      CALL TIME3(IT,IY,IM,IDD,IH,IX,ISEC,ICEN) 
      NUMDM1=ID(IM) 

      IF(IM.EQ.2.AND.IY.EQ.4*(IY/4)
     .	.AND.MOD(IY,100).NE.0)  NUMDM1=29	! Centuries are not leap yrs

      IF(IM.EQ.2.AND.IY.EQ.400*(IY/400)) 
     .				NUMDM1=29	! Every 400 years is tho...
      

      RETURN 
      END 
