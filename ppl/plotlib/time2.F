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
*	Mod 11.20.95 J Davison to accommodate 4 digit year, and seconds
*       J Davison 10.15.99   Modified to support multiple calendars.
* V530 1/01 *acm* split calendar.cmn into .cmn and .decl

      CHARACTER ITIME*(*),SYM*(*)
C                                                                        

        include 'calendar.decl' 
        include 'calendar.cmn'

C     THIS ROUTINE TRANSLATES enhanced WOODSHOLE FORMATTED TIME
C     INTO A PLOTABLE TYPE FORMAT.                                       
C         IE.  77081904590118 BECOMES 4:59:01 AUG 19, 1877
C     IN THE GREGORIAN CALENDAR, WHERE MONTH 8 IS 'AUG'. THE VALUES OF
C     MONTH_NAMES IN THE CURRENT CALENDAR DETERMINES THE MONTH NAMES.
C                                                                        

**************************************************************************

      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)                   

*     WRITE(SYM,40) IHR,IMIN,ISEC,MONTH_NAMES(IMON),IDAY,ICEN,IYR         
*  40 FORMAT(I2,':',I2.2,':',I2.2,' ',A3,' ',I2,',', I2.2, I2.2)
 
      WRITE(SYM,40) IHR,IMIN,MONTH_NAMES(IMON),IDAY,ICEN,IYR         

   40 FORMAT(I2,':',I2.2,' ',A3,' ',I2,',', I2.2, I2.2)
      RETURN                                                             
      END                                                                
