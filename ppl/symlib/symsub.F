	SUBROUTINE SYMSUB(LINE,ILEN,IER,SYM,IPOS)
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
C**    @(#)symsub.f	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL SYMSUB(LINE,ILEN,IER,SYM)
C**
C**	Parameters:
C**		LINE -- String to have symbols substituted (char*2048)
C**		ILEN -- length of LINE before and after substitution
C**		IER -- error code
C**			0 = success
C**			1 = symbol not found
C**			2 = unmatched '
C**		SYM -- Symbol not found
C**		IPOS -- position of error
C**
C**	Description:
C**		Substitutes the values of symbols found in LINE.  Only
C**		symbols of the form 'sym' are substitued.  A double '
C**		is enterpretated as a single '.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C* v552 *acm* 3/03 Longer strings: changes for multi-line labels.

	CHARACTER LINE*(*),STR*2048,SYM*(*),GET*2048
	IER=0
	INXT=1
100	IST=INDEX(LINE(INXT:),'''')+INXT-1
	IF(IST.EQ.INXT-1.OR.IST.GT.ILEN)RETURN
	STR=LINE(:IST-1)
	ISP=INDEX(LINE(IST+1:),'''')+IST
	IF(ISP.EQ.IST)THEN
C	    UNMATCHED '
	    IER=2
	    IPOS=IST
	    RETURN
	ELSE IF((ISP-IST).EQ.1)THEN
C	    DOUBLE '
	    STR(IST:)=LINE(ISP:ILEN)
	    ILEN=ILEN-1
	    INXT=IST+1
	ELSE
C	    EXTRACT SYMBOL AND SUBSTITUTE
	    SYM=LINE(IST+1:ISP-1)
	    CALL GETSYM(SYM,GET,NC,IER)
	    IF(IER.EQ.0)THEN
		STR(IST:)=GET(:NC)
		STR(IST+NC:)=LINE(ISP+1:ILEN)
		ILEN=ILEN-(ISP-IST+1)+NC
		INXT=IST
	    ELSE
		IPOS=IST+1
		RETURN
	    ENDIF
	ENDIF
	LINE=STR(1:ILEN)
	GOTO 100
	END
