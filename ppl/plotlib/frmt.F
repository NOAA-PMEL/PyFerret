      SUBROUTINE FRMT(XMIN,XMAX,IPRES,IF) 
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
C**    @(#)frmt.f	1.1    3/10/88
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
      CHARACTER*(*) IF 
C     THIS SUBROUTINE CHOOSE THE FORMAT   NOT ANSI STANDARD 
      DX=ABS(XMAX-XMIN) 
      XX=ABS(XMAX) 
      IF(XX.LT.ABS(XMIN))XX=ABS(XMIN) 
      IF(XX.GE.1000..OR.XX.LT..1)GOTO 100 
200   IP=ABS(IPRES) 
      ID=IP-IFIX(ALOG10(DX)+.5) 
      IF(ID.LT.0)ID=0 
      IB=IFIX(ALOG10(XX*1.00001)) 
      WRITE(IF,999)IB+ID+3,ID 
999   FORMAT('(F',I2,'.',I1,')') 
      RETURN 
100   IF(IPRES.LT.0)GOTO 200 
      ID=IPRES-1 
      IF(ID.LT.0)ID=0 
      WRITE(IF,998) ID+7,ID 
998   FORMAT('(1PE',I2,'.',I1,')') 
      RETURN 
      END 
