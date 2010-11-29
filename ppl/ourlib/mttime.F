      SUBROUTINE MTDAT(IT,STR)
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
C**    @(#)mttime.F	1.1    3/10/88
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
	REAL IT(3)
      CHARACTER STR*(*),MONTH(12)*3
      DATA MONTH/'JAN','FEB','MAR','APR','MAY','JUN',
     *           'JUL','AUG','SEP','OCT','NOV','DEC'/
      CALL MTMDY(IT,IM,ID,IY)
      IH=IT(3)
      WRITE(STR,999)MONTH(IM),ID,1900+IY,IH
999   FORMAT(A3,I3,',',I4,I6.4)
      RETURN
      END
      SUBROUTINE MTMDY(IT,IM,ID,IY)
	REAL IT(3)
      DIMENSION IMN(12)
      DATA IMN/1,32,60,91,121,152,182,213,244,274,305,335/
      IY=IT(1)
      IJ=IT(2)
      ITST=1
      IF(IY-(IY/4)*4.EQ.0)ITST=0
      IF(IJ.LT.61-ITST)THEN
      IM=IJ/32+1
      ID=IJ-IMN(IM)+1
      ELSE
      IM=INT((IJ+ITST)/30.6001+.02)+1
      ID=IJ-IMN(IM)+ITST
      END IF
      RETURN
      ENTRY MDYMT(IT,IM,ID,IY)
      IT(1)=IY
      IF(IY-(IY/4)*4.EQ.0)THEN
      IF(IM.LT.3)THEN
      IT(2)=IMN(IM)+ID-1
      ELSE
      IT(2)=IMN(IM)+ID
      END IF
      ELSE
      IT(2)=IMN(IM)+ID-1
      END IF
      RETURN
      END
      INTEGER*4 FUNCTION MTMIN(MTIME)
C     REVISED VERSION THAT WILL WORK FOR LEAP YEARS
C     MINUTES SINCE 1-1-1970
	REAL MTIME(3)
	INTEGER*4 JDN
      CALL MTMDY(MTIME,IM,ID,IY)
      JH=MTIME(3)/100
      JM=AMOD(MTIME(3),100.)
      IF(IM.GT.2)THEN
      IM=IM+1
      ELSE
      IY=IY-1
      IM=IM+13
      END IF
#ifdef unix
      JDN=INT(365.25*IY)+INT(30.6001*IM)+ID-25631
#else
      JDN=JINT(365.25*IY)+JINT(30.6001*IM)+ID-25631
#endif
      MTMIN=JDN*1440+JH*60+JM
      RETURN
      END
      SUBROUTINE MINMT(NMS70,IT)
C     REVISED VERSION THAT DOES LEAP YEAR CORRECTLY
	REAL IT(3)
	INTEGER*4 NMS70,JDN
      JDN=(NMS70/1440)+25631
      IY=INT((JDN-122.1)/365.25)
#ifdef unix
      IM=INT((JDN-INT(365.25*IY))/30.6001)
      ID=JDN-INT(365.25*IY)-INT(30.6001*IM)
#else
      IM=INT((JDN-JINT(365.25*IY))/30.6001)
      ID=JDN-JINT(365.25*IY)-JINT(30.6001*IM)
#endif
      IF(IM.LT.14)THEN
      IM=IM-1
      ELSE
      IM=IM-13
      IY=IY+1
      END IF
      CALL MDYMT(IT,IM,ID,IY)
      LEFT=NMS70-(NMS70/1440)*1440
      IH=(LEFT/60)
      IM=LEFT-60*IH
      IT(3)=IH*100+IM
      RETURN
      END
