      SUBROUTINE TIME1(I1,ITIME1,I2,ITIME2,I3,ITIME3,DT)               
C** 
C**    @(#)time1.F	1.1    3/10/88
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
C     THIS ROUTINE DOES TIME CALCULATIONS.                             
C         I1 = NUMBER OF A POINT IN OUR SERIES FOR WHICH THE TIME      
C              OF OCCURRENCE OS KNOWN.                                 
C         ITIME1 = THE KNOWN TIME OF OCCURRENCE IN WOODSHOLE FORMAT.   
C         I2,ITIME2 = CONTROL VARIBLES.                                
C             ONE OF THESE MUST BE 0.  THE OTHER MUST BE >= TO THE     
C             THE CORRESPONDING 1 VARIBLE.  IF I2>0 I3=I2 AND          
C             ITIME3= TIME PT I2 OCCURS.  IF ITIME2 .NE.0, I2=0,I3     
C             =POINT TIME ITIME2 OCCURS, AND ITIME3 = EXACT TIME POINT 
C             I3 OCCURS.                                               
C         DT = SAMPLING RATE IN SECONDS.                               
C                                                                      
C     THIS PROGRAM WAS MODIFIFIED IN JUNE,1980.  AT THAT TIME WE       
C     TRYED TO IMPROVE THE EFFICIENCY OF THIS PROGRAM.                 
C                                                                      
C     SETTING DEFAULTS.                                                
	INTEGER*4 JTIME1,JTIME2,JDAY
	CHARACTER*10 ITIME1,ITIME2,ITIME3
C
C	FIND POINT AT WHICH ITIME2 OCCURS
C
	IF(I2.EQ.0)THEN
	CALL TIME3(ITIME1,IYR,IMON,IDAY,IHR,IMIN)
	JTIME1=JDAY(IYR,IMON,IDAY,IHR,IMIN)
	CALL TIME3(ITIME2,IYR,IMON,IDAY,IHR,IMIN)
	JTIME2=JDAY(IYR,IMON,IDAY,IHR,IMIN)
	I3=FLOAT(JTIME2-JTIME1)/DT+.5
#ifdef unix
	JTIME2=INT(FLOAT(I3)*DT)+JTIME1
#else
	JTIME2=JINT(FLOAT(I3)*DT)+JTIME1
#endif
	I3=I3+I1
	CALL DAYJ(JTIME2,IYR,IMON,IDAY,IHR,IMIN)
	WRITE(ITIME3,999)IYR,IMON,IDAY,IHR,IMIN
	ELSE
30	I3=I2
C
C	FIND ITIME3 BY POINT I2.
C
	CALL TIME3(ITIME1,IYR,IMON,IDAY,IHR,IMIN)
	JTIME1=JDAY(IYR,IMON,IDAY,IHR,IMIN)
#ifdef unix
	JTIME2=JTIME1+INT(FLOAT(I3-I1)*DT)
#else
	JTIME2=JTIME1+JINT(FLOAT(I3-I1)*DT)
#endif
	CALL DAYJ(JTIME2,IYR,IMON,IDAY,IHR,IMIN)
	WRITE(ITIME3,999)IYR,IMON,IDAY,IHR,IMIN
999	FORMAT(5I2.2)
	ENDIF
	RETURN
	END
	SUBROUTINE DAYJ(JDAY,IY,IMN,ID,IH,IM)
	INTEGER*4 JDAY,JDN
	JDN=JDAY/1440
	IY=INT((JDN-122.1)/365.25)
#ifdef unix
	IMN=INT((JDN-INT(365.25*IY))/30.6001)
	ID=JDN-INT(365.25*IY)-INT(30.6001*IMN)
#else
	IMN=INT((JDN-JINT(365.25*IY))/30.6001)
	ID=JDN-JINT(365.25*IY)-JINT(30.6001*IMN)
#endif
	IF(IMN.LT.14)THEN
	IMN=IMN-1
	ELSE
	IMN=IMN-13
	IY=IY+1
	ENDIF
	LEFT=JDAY-(JDAY/1440)*1440
	IH=(LEFT/60)
	IM=LEFT-60*IH
	RETURN
	END
