      SUBROUTINE TIME2(ITIME,SYM)                                    
C** 
C**    @(#)time2.f	1.1    3/10/88
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


      CHARACTER IMN(12)*3,ITIME*(*),SYM*(*)
      DATA IMN/'JAN','FEB','MAR','APR','MAY','JUN', 
     *'JUL','AUG','SEP','OCT','NOV','DEC'/ 
C                                                                        
C     THIS ROUTINE TRANSLATES enhanced WOODSHOLE FORMATTED TIME          
         
C     INTO A PLOTABLE TYPE FORMAT.                                       
C         IE.  77081904590118 BECOMES 4:59:01 AUG 19, 1877
                     
C                                                                        
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)                   

*     WRITE(SYM,40) IHR,IMIN,ISEC,IMN(IMON),IDAY,ICEN,IYR         
*  40 FORMAT(I2,':',I2.2,':',I2.2,' ',A3,' ',I2,',', I2.2, I2.2)
 
      WRITE(SYM,40) IHR,IMIN,IMN(IMON),IDAY,ICEN,IYR         

   40 FORMAT(I2,':',I2.2,' ',A3,' ',I2,',', I2.2, I2.2)
      RETURN                                                             
      END                                                                
