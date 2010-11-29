	SUBROUTINE DAYWK1(ITIME,DAY)
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
