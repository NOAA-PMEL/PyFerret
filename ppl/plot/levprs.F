	SUBROUTINE LEVPRS(PRSE,VAL,INTIF,IER,IPOS)
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
C**    @(#)levprs.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*	*jd* MOD 3.29.93 READ fails on SGI with comma in string.  Eliminate.
*	*jd* MOD 11.1.94 READ fails on IBM AIX without it.  Put back in.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	LOGICAL BLNK
	CHARACTER PRSE*(*)

*       variable IF renamed to INTIF for MAC *jd* 7.92

	DIMENSION VAL(4),INTIF(4)
	IER=0
	IPOS=1
	I=1
30	ICM=INDEX(PRSE,',')
	IF(ICM.EQ.0)ICM=256
	IBL=INDEX(PRSE,' ')
	IF(IBL.EQ.0)IBL=256
	ISTPP=MIN0(ICM,IBL,255)
	BLNK=PRSE(ISTPP:ISTPP).EQ.' '
	IF(ISTPP.NE.1)THEN
		INTIF(I)=1
		PRSE(ISTPP:ISTPP)=','

#ifdef AIX_XLF
		READ(PRSE(:ISTPP),996,ERR=1000)VAL(I)
#else
		READ(PRSE(:ISTPP-1),996,ERR=1000)VAL(I)
#endif
996		FORMAT(E20.0)
	ENDIF
20	ISTPP=ISTPP+1
	IF(ISTPP.GT.255)GOTO 40
	IF(PRSE(ISTPP:ISTPP).EQ.' '.OR.
     *	(BLNK.AND.PRSE(ISTPP:ISTPP).EQ.','))GOTO 20
	PRSE=PRSE(ISTPP:)
	IPOS=IPOS+ISTPP-1
	I=I+1
	IF(I.LE.4)GOTO 30
40	RETURN
1000	IER=9
	RETURN
	END
