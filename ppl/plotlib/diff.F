	DOUBLE PRECISION FUNCTION DIFF(IT1,IT2)
C** 
C**    @(#)diff.F	1.1    3/10/88
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
*	Mod J Davison 11.20.95 for 4 digit year and seconds

	CHARACTER*(*) IT1,IT2
*  *kob*     - Need the below to stop OSF/Linux crashing of v5.4 - w/ pplldx envelopes
*              and c/fortran argument passing, need exact length strings to 
*              pass into whoi2bc
*  *kob* 1/10/02 - for some reason, also need to explicitely only use first 14
*                  chars of IT1, IT2.  This only showed up as a problem with Linux
*                  linked statically, for some odd reason
* V695  *acm*  9/15 Changes for hour:min labeling on time axes

	CHARACTER*14 IT1_14, IT2_14
	INTEGER*4 JDAY
	REAL*8 DATE2, DATE1, WHOI2BC, RDATE
	
	IT1_14 = IT1(:14)
	IT2_14 = IT2(:14)
	RDATE = WHOI2BC(IT2_14) - WHOI2BC(IT1_14)
	DIFF  = RDATE

*       Original code
*	CALL TIME3(IT1,IY1,IM1,ID1,IH1,IMN1,ISC1,ICN1)
*	CALL TIME3(IT2,IY2,IM2,ID2,IH2,IMN2,ISC2,ICN2)

*	DIFF=DOUBLE(JDAY(IY2,IM2,ID2,IH2,IMN2,ISC2,ICN2)-
*     * JDAY(IY1,IM1,ID1,IH1,IMN1,ISC1,ICN1))

	RETURN
	END
