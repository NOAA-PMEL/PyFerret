	SUBROUTINE ZABMV
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
C**    @(#)zabmv.f	1.1    3/10/88
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

C	CALL ZABMV
C
C	This routine creates a binary file that can be used to
c	construct device dependent plotting code
c
c	the file format is:
c
c	dimension x(63),y(63)
c	write(luhp)x,y
c
c	where:
c			x>0, y>0	is the coordinate to be
c					moved to, pen down
c
c			x<0, y>0	is a pen up move to -x,y
c
c			x>0, y<0	select pen x
c
c			x<0, y<0	plot end
c
c	D. Denbo  February 1985
c
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	IF(.NOT.PLTFLG)THEN
	CALL BINBUF(ASIZE,BSIZE)
C
C	SELECT PEN 1
C
	CALL BINBUF(1.,-1.)
	PLTFLG=.TRUE.
	LPEN=.NOT.PEN
	ENDIF
	IF(PEN)THEN
	CALL BINBUF(A,B)
	ELSE
	CALL BINBUF(-A-.001,B)
	ENDIF
	RETURN
	END
	SUBROUTINE BINBUF(XN,YN)
C
C	fills the file buffer with a flush function
C
	DIMENSION XB(63),YB(63)
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
#ifdef unix
	CHARACTER ZFILE*81
#endif
	IF(NEWFIL)THEN
#ifdef unix
	len=lnblk(filez,81)
	icnt=1
100	write(zfile,999)filez(:len),fzcnt
999	format(a,i3.3)
	fzcnt=fzcnt+1
	icnt=icnt+1
	if(icnt.gt.100)stop 'can not open meta file'
	OPEN(UNIT=LUHP,FILE=ZFILE,FORM='UNFORMATTED',
     *	STATUS='NEW',ERR=100)
#else
	OPEN(UNIT=LUHP,FILE=FILEz,FORM='UNFORMATTED',
     *	STATUS='NEW')
#endif
	NEWFIL=.FALSE.
	IPT=1
	ENDIF
	XB(IPT)=XN
	YB(IPT)=YN
	IPT=IPT+1
	IF(IPT.GT.63)THEN
	WRITE(LUHP)XB,YB
	IPT=1
	ENDIF
	RETURN
	ENTRY BINFSH
	XB(IPT)=-1.
	YB(IPT)=-1.
	WRITE(LUHP)XB,YB
	CLOSE(LUHP)
	NEWFIL=.TRUE.
	IPT=1
	RETURN
	END
