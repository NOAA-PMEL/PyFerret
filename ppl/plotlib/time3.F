      SUBROUTINE TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)              
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
*	Mod 11.20.95 J Davison to accommodate 4 digit year, and seconds


C                                                                        
C     THIS ROUTINE TAKES A GIVEN TIME IN WOODSHOLE FORMAT                
C     AND BREAKS IT IN TO THE SECOND, MIN, HOUR, DAY, MONTH,             
C     AND YEAR.                                                          
C                                                                        
	CHARACTER ITIME*(*)
	READ(ITIME,'(7I2)') IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN

      RETURN                                                             
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **                              
      END                                                                
