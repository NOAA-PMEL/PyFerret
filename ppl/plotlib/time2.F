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
      CHARACTER IMN(12)*3,ITIME*10,SYM*17 
      DATA IMN/'JAN','FEB','MAR','APR','MAY','JUN', 
     *'JUL','AUG','SEP','OCT','NOV','DEC'/ 
C                                                                        
C     THIS ROUTINE TRANSLATES WOODSHOLE FORMATTED TIME                   
C     INTO A PLOTABLE TYPE FORMAT.                                       
C         IE.  770819045903 BECOMES 4:59 AUG 19,1977                     
C                                                                        
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN)                   
      WRITE(SYM,40) IHR,IMIN,IMN(IMON),IDAY,IYR         
   40 FORMAT(I2,':',I2.2,' ',A3,' ',I2,',19',I2.2)
      RETURN                                                             
      END                                                                
