	SUBROUTINE CHIN(IN,NCH,ITERM)
C** 
C**    @(#)chin.F	1.1    3/10/88
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
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
#ifdef unix
	integer IN(NCH)
	character inc*255
C
	IF(FILET(1:8).NE.'/dev/tty')RETURN
	read(*,999)inc
999	format(a)
	do 10 i=1,nch
	in(i)=ichar(inc(i:i))
10	continue
#else
        BYTE IN(NCH),IOSB(4)
        DIMENSION IPAR(6)
        include '($IODEF)'
C
        IF(FILET(1:3).NE.'TI:'.and.FILET(1:3).NE.'TT:')RETURN
        CALL SYS$ASSIGN('TT',CHAN,,)
	CALL SYS$QIOW(,%VAL(CHAN),%VAL(IO$_TTYREADALL .OR. IO$M_NOECHO),,,,
     *	IN,%VAL(NCH),,,,)
        CALL SYS$DASSGN(%VAL(CHAN))
#endif
	RETURN
	END
