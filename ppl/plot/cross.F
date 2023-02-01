	SUBROUTINE CROSS
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
C**    @(#)cross.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'AXIS.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'LINES.INC'

	IF(ICROSS.EQ.1)THEN
	    IF(XLO.LT.0..AND.XHI.GT.0.)THEN
		CALL PLOT(0.,YLO,0,0)
		CALL PLOT(0.,YHI,1,0)
	    ENDIF
	    IF(YLO.LT.0..AND.YHI.GT.0.)THEN
		CALL PLOT(XLO,0.,0,0)
		CALL PLOT(XHI,0.,1,0)
	    ENDIF
	ELSE IF(ICROSS.NE.0)THEN
	    IF(ICROSS.EQ.2.OR.ICROSS.EQ.4)THEN
		DO 90 I=1,LNUM
		DO 95 J=I+1,LNUM
		IF(YOFF(I).EQ.YOFF(J))GOTO90
95		CONTINUE
		CALL PLOT (XLO,YOFF(I),0,0)
		CALL PLOT (XHI,YOFF(I),1,0)
90		CONTINUE
	    ENDIF
	    IF(ICROSS.GE.3)THEN
		DO 100 I=1,LNUM
		DO 105 J=I+1,LNUM
		IF(XOFF(I).EQ.XOFF(J))GOTO 100
105		CONTINUE
		CALL PLOT(XOFF(I),YLO,0,0)
		CALL PLOT(XOFF(I),YHI,1,0)
100		CONTINUE
	    ENDIF
	ENDIF
	RETURN
	END
