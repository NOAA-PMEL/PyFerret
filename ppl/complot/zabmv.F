	SUBROUTINE ZABMV
C** 
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
