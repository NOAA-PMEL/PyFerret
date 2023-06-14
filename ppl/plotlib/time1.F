      SUBROUTINE TIME1(I1,ITIME1,I2,ITIME2,I3,ITIME3,DT)               
C start of FERRET mod
C given a time, itime1, in WHOI format (ie, a date) and its corresponding time
C step value, i1 - compute either the date (itime3)  corresponding to another
C time step (i2) or the time step (i3) corresponding to another date (itime2)
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* note DAYJ is not included in this module although it is in the TIME1.FOR file
* from PPLUS :
C	Directory PMEL:[PLOT5.LIBRARY.MILELIB]
C	TIME1.FOR;2               5   2-FEB-1985 10:13 
C search for "FERRET" to find all revisions
C *sh* 9/23/88 - rev 1.0 for FERRET 1.10+
C - eliminated JDAY and DAYJ which convert dates to and from minutes since
C	  1970 ("J"="JULIAN") in favor of WHOI2BC and BC2WHOI based on Jan.1, 0
C	  to work around JDAY bug with the year "00"
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C explicit declarations:
      INTEGER		i1, i2, i3
      INTEGER		iyr, imon, iday, ihr, imin
      REAL		dt
      REAL*8		WHOI2BC, t1, t2
      CHARACTER*14      BC2WHOI
C end of FERRET mods 9-22-88

*	Mod J Davison 11.21.95 For 4 digit year

                                           
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
                                                                      
C     THIS PROGRAM WAS MODIFIFIED IN JUNE,1980.  AT THAT TIME WE       
C     TRYED TO IMPROVE THE EFFICIENCY OF THIS PROGRAM.                 
                                                                      
C     SETTING DEFAULTS.                                                
      INTEGER*4 JTIME1,JT,JTIME2,JDAY
      CHARACTER*(*) ITIME1,ITIME2,ITIME3
      CHARACTER*14 ITIME

C	FIND POINT AT WHICH ITIME2 OCCURS

      IF(I2.EQ.0)THEN
      CALL  TIME3(ITIME1,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      JTIME1=JDAY(IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      CALL TIME3(ITIME2,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      JTIME2=JDAY(IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      I3=FLOAT(JTIME2-JTIME1)/DT+.5
      JTIME2=INT(FLOAT(I3)*DT)+JTIME1
      I3=I3+I1
      CALL  DAYJ(JTIME2,IYR,IMON,IDAY,IHR,IMIN)
      WRITE(ITIME3,999)IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN
      ELSE
30	I3=I2

C	FIND ITIME3 BY POINT I2.

C FERRET mod 9/23/88 *sh*
C original code:
C	CALL TIME3(ITIME1,IYR,IMON,IDAY,IHR,IMIN)
C	JTIME1=JDAY(IYR,IMON,IDAY,IHR,IMIN)
C	JTIME2=JTIME1+JINT(FLOAT(I3-I1)*DT)
C	CALL DAYJ(JTIME2,IYR,IMON,IDAY,IHR,IMIN)
C	WRITE(ITIME3,999)IYR,IMON,IDAY,IHR,IMIN
C new code:
        T1     = WHOI2BC( itime1 )
        T2     = T1 + DBLE( (I3-I1) * DT )
        ITIME3 = BC2WHOI( T2 )
C end of FERRET mod 9/23/88 *sh*
999	FORMAT(7I2.2)
      ENDIF
      RETURN
      END

*     *jd* 3.11.92 added subroutine DAYJ from plotlib/time1.f
*     To prevent linking warnings 
        SUBROUTINE DAYJ(JDAY,IY,IMN,ID,IH,IM)
        INTEGER*4 JDAY,JDN
        JDN=JDAY/1440
        IY=INT((JDN-122.1)/365.25)
        IMN=INT((JDN-INT(365.25*IY))/30.6001)
        ID=JDN-INT(365.25*IY)-INT(30.6001*IMN)
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


