      SUBROUTINE TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN)              
C** 
C**    @(#)time3.f	1.1    3/10/88
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
C     THIS ROUTINE TAKES A GIVEN TIME IN WOODSHOLE FORMAT                
C     AND BREAKS IT IN TO THE SECOND, MIN, HOUR, DAY, MONTH,             
C     AND YEAR.                                                          
C                                                                        
	CHARACTER ITIME*10
	READ(ITIME,'(5I2)')IYR,IMON,IDAY,IHR,IMIN
      RETURN                                                             
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **                              
      END                                                                
