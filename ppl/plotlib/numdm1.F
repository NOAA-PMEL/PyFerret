      FUNCTION NUMDM1(IT) 
C** 
C**    @(#)numdm1.f	1.1    3/10/88
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
* 	Mod J Davison 11.22.95 for 4 digit years
* 	Mod J Davison  2.6.96  fix bug re 400 year biz
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

C     THIS FUNCTION RETURNS NO. OF DAYS IN CURRENT MONTH
 	CHARACTER*(*) IT
      DIMENSION ID(12) 
      DATA ID/31,28,31,30,31,30,31,31,30,31,30,31/ 
      CALL TIME3(IT,IY,IM,IDD,IH,IX,ISEC,ICEN) 
      NUMDM1=ID(IM) 

      IF(IM.EQ.2.AND.IY.EQ.4*(IY/4)
     .	.AND.MOD(IY,100).NE.0)  NUMDM1=29	! Centuries are not leap yrs

      IF(IM.EQ.2.AND.IY.EQ.400*(IY/400)) 
     .				NUMDM1=29	! Every 400 years is tho...
      

      RETURN 
      END 
