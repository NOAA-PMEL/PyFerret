	SUBROUTINE LEVSRT
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
C**    @(#)levsrt.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/CONT.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:CONT.INC'
#endif
	LOGICAL BLNK
300	BLNK=.FALSE.
	DO 310 I=2,NLEV
	IF(ZLEV(I-1).GT.ZLEV(I))THEN
		BLNK=.TRUE.
		Z=ZLEV(I)
		J=LDIG(I)
		K=LWGT(I)
		JC=LCLR(I)
		ZLEV(I)=ZLEV(I-1)
		LDIG(I)=LDIG(I-1)
		LWGT(I)=LWGT(I-1)
		LCLR(I)=LCLR(I-1)
		ZLEV(I-1)=Z
		LDIG(I-1)=J
		LWGT(I-1)=K
		LCLR(I-1)=JC
	ENDIF
310	CONTINUE
	IF(BLNK)GOTO 300
	RETURN
	END
