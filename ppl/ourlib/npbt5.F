      FUNCTION NPBT(MTIME2,MTIME1,NMBP,JFLAG)
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
C**    @(#)npbt5.f	1.1    3/10/88
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
C         NPBT RETURNS THE NUMBER OF DATA POINTS BETWEEN TWO
C         MILITARY TIMES, GIVEN THE SAMPLING RATE IN MINUTES.
C
C         MTIME2: THE LATER OF THE TWO TIMES.
C         MTIME1: THE EARLIER OF THE TWO TIMES.
C         NMBP:   THE NUMBER OF MINUTES BETWEEN POINTS(SAMPLING RATE).
C         JFLAG:  0: COMPUTE THE NUMBER OF WHOLE SAMPLE INTERVALS
C                 BETWEEN THE TIMES.
C                 1: COMPUTE THE NUMBER OF WHOLE SAMPLE INTERVALS
C                 THAT INCLUDES THE TIMES.

C
	REAL MTIME1(3),MTIME2(3)
	INTEGER*4 MTMIN,MNDF
      MNDF=MTMIN(MTIME2)-MTMIN(MTIME1)
      NPBT=MNDF/NMBP
      IF(JFLAG.EQ.1) NPBT=NPBT+1
      RETURN
      END
