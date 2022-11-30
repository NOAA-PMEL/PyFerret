      SUBROUTINE HABIS(IPLOT,IPAUSE,IHDCPY,IN)
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
C**    @(#)habis.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INCLUDE 'PLTCOM.DAT'
	INCLUDE 'SYSTEM.INC'
	CHARACTER INS*1
C
C     THIS ROUTINE IS OUR ALTERNATIVE TO PLOTEND.
C         IPLOT=XXXX0YYYY0Z
C             XXXX=BAUD
C             YYYY=TEK MODEL
C             Z=ITYPE (PLOT TYPE)
C         IPAUSE = 0 FOR NO PAUSE
C                = 1 FOR A  PAUSE
C         IHDCPY= 0 FOR NO HARDCOPY
C                = 1 FOR A HARD COPY
C
	CALL ALPHAS
	CALL PAGE
	IF(IHDCPY.EQ.1) CALL HDCOPY(IPLOT)
	IF(IPAUSE.EQ.1.AND.(PTYPE.EQ.1.OR.IABS(PTYPE).GE.2)) THEN
	    CALL CHOUT(BEL,1)
	    CALL CHOUT(BEL,1)
	    CALL CHDMP
	    IF(PTYPE.GE.3)THEN
		OPEN(99,FILE='/dev/tty',STATUS='UNKNOWN')
 99		READ(99,'(A1)',ERR=99,END=100)INS
100		IN=ICHAR(INS)
	    ELSE
		CALL CHIN(IN,1,0)
	    ENDIF
	ENDIF
	call clsdev(termf,sover)
	RETURN
	END
