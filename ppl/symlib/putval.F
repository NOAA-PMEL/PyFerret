	SUBROUTINE PUTVAL(STR,X,IPRES,IER)
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
C**    @(#)putval.f	1.2    5/26/88
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
C**		CALL PUTVAL(STR,X,IPRES,IER)
C**
C**	Parameters:
C**		STR -- Symbol name (char*30)
C**		X -- Real value of symbol
C**		IPRES -- number of digits, i.e, 1PG10.IPRES
C**		IER -- error code
C**			0 = sucess
C**			1 = error other than symbol not found
C**
C**	Description:
C**		Updates or creates the value for symbol STR.
C**
C**	History:
C**	    Written:
C**		14-JAN-1987	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C * v552 *acm* 3/03 Longer strings: changes for multi-line labels

	CHARACTER STR*(*),VAL*2048,FRMT*30
	IP=IPRES
	IF(IPRES.GT.10.OR.IPRES.LT.0)IP=4
	IL=IP+7
	WRITE(FRMT,999)IL,IP
999	FORMAT('(1PG',I3.3,'.',I3.3,')')
	WRITE(VAL,FRMT)X
100	IF(VAL(1:1).EQ.' ')THEN
	    VAL=VAL(2:)
	    IL=IL-1
	    GOTO 100
	ENDIF
	NC=LNBLK(VAL,IL)
	CALL PUTSYM(STR,VAL,NC,IER)
	RETURN
	END
