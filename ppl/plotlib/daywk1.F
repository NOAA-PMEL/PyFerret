	SUBROUTINE DAYWK1(ITIME,DAY)
C** 
C**    @(#)daywk1.f	1.1    3/10/88
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

	CHARACTER ITIME*(*),WEEK(8)*3,DAY*(*)
      DATA WEEK/'SUN','MON','TUE','WED','THU','FRI','SAT','SUN'/         
C                                                                        
C     THIS FUNCTION RETURNS THE FIRST THREE LETTERS OF THE DAY OF        
C     THE WEEK, GIVEN A TIME IN WOODSHOLE FORMAT.                        
C     IE. (000101000000)=3HMON                                     
C     ALGORITHM TAKEN FROM HP-19C/HP-29C APPLICATIONS BOOK.              
C                                                                        
C                                                                        
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN)                      
      IYP=IYR+1899                                                       
      IF(IMON.GT.2) IYP=IYR+1900                                         
      IMP=IMON+13                                                        
      IF(IMON.GT.2) IMP=IMON+1                                           
      NDAY=IFIX(365.25*FLOAT(IYP))+IFIX(30.6001*FLOAT(IMP))+IDAY         
      NDAY=MOD(NDAY+5,7)                                                 
      DAY=WEEK(NDAY+1)                                                
C     RECALL 1900 IS NOT A LEAP YAER.                                    
      IF(ITIME.LT.'0003010000') DAY=WEEK(NDAY+2)                      
      RETURN                                                             
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **                              
      END                                                                
