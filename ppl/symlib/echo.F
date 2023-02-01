	SUBROUTINE ECHO(LINE,ILEN)
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
C**    @(#)echo.F	1.3    5/31/88
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
C**		CALL ECHO(LINE,ILEN)
C**
C**	Parameters:
C**		LINE -- Command string to be echoed.
C**		ILEN -- Length of LINE
C**
C**	Description:
C**		Writes LINE on the current ECHO file.  If LINE is too
C**		long a continuation line is also written.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
	include 'CMDCOM.INC'

C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER LINE*(*),LEV*5
	LOGICAL LEVF
	LEVF=CMDLEV.GT.1
	IF(LEVF)THEN
	    WRITE(LEV,999)CMDLEV
999	    FORMAT('C ',I1,'- ')
	ENDIF
	K=1
200	KS=MIN0(K+69,ILEN)
	IF(KS.NE.ILEN)THEN
	    IF(LEVF)THEN
		WRITE(ECHLUN,998)LEV,LINE(K:KS)
998		FORMAT(A5,A70,'-')
	    ELSE
		WRITE(ECHLUN,996)LINE(K:KS)
996		FORMAT(A70,'-')
	    ENDIF
	ELSE
	    IF(LEVF)THEN
		WRITE(ECHLUN,997)LEV,LINE(K:ILEN)
997		FORMAT(A5,A)
	    ELSE
		WRITE(ECHLUN,995)LINE(K:ILEN)
995		FORMAT(A)
	    ENDIF
	    RETURN
	ENDIF
	K=KS+1
	GOTO 200
	END
