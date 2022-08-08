	SUBROUTINE LEVSET(IST,VAL,INTIF,LWGH,NL)
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
C**    @(#)levset.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

* V510 *sh* 3/00 - bug fix - "DEL" ignored because of misspelling of lwgh
* V62+ *acm 6/09 - bug fix #1672 too few levels computed depending on the values
* V62+ *acm 6/09 - rename variable IF to INTIF
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V631 *acm* 10/09 For the fix to bug 1672(present only in 32-bit linux), the
*                  comparison must be done with TM_FPEQ. This was the cause of bug 1707.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V741 *acm* 6/18  issue 1877, extra unwanted level was added
* V767 *acm* 6/22 Cosmetic changes, looking into PyFerret issue 116

	INCLUDE 'CMRDL.INC'

	DIMENSION VAL(4),INTIF(4)
	LOGICAL TM_FPEQ_EPS_SNGL
	REAL top

	LWGH=0
	IF(INDEX(LABEL(:IST),'DEL').NE.0)LWGH=5
	IF(INDEX(LABEL(:IST),'PEN').NE.0)LWGH=4
	IF(INDEX(LABEL(:IST),'DASH').NE.0)LWGH=3
	IF(INDEX(LABEL(:IST),'DARK').NE.0)LWGH=2
	IF(INDEX(LABEL(:IST),'LINE').NE.0)LWGH=1
	IF(INTIF(2).EQ.0)THEN
	   NL=1
	ELSE
	   IF(INTIF(3).EQ.0)THEN
	      VAL(3)=VAL(2)-VAL(1)
	      NL=2
	   ELSE

	      IF(VAL(3).NE.0.)THEN
		 NL = INT((VAL(2)-VAL(1))/VAL(3))+1
		 top = VAL(1)+(NL-1)*VAL(3)

* Add another level if neeeded: do not add iI=f the top level given and 
* computed top level differ by1% or less

		 IF (top .LT. VAL(2)) THEN 
		    IF ( .NOT.TM_FPEQ_EPS_SNGL(0.01, val(2), top) ) NL = NL + 1 
	         ENDIF
	      ELSE
		 NL=2
	      ENDIF

	   ENDIF
	ENDIF
	RETURN
	END
