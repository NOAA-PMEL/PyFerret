	SUBROUTINE CHOUT(CHR,ICNT)
C** 
C**    @(#)chout.F	1.3    5/25/88
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
	integer chr(icnt)
	character*255 buf
#else
	byte iosb(4),buf(80),chr(icnt)
	include '($IODEF)'

#endif
C
C	WRITES A SINGLE CHARACTER TO THE TERMINAL
C	D. DENBO JUNE 1983  UPDATED MARCH 1984
C
	DATA IPT/1/
	J=1
#ifdef unix
40	BUF(IPT:IPT)=CHAR(CHR(j))
#else
40	buf(ipt)=chr(j)
#endif
	IPT=IPT+1
	J=J+1
	IF(IPT.GT.80)THEN
C
#ifdef unix
	IF(FILET(1:8).EQ.'/dev/tty')THEN
	    do 50 i=1,80
		istat=putch(buf(i:i))
50	    continue
#else
        IF(FILET(1:3).EQ.'TT:')THEN
            CALL SYS$ASSIGN('TT',CHAN,,)
            CALL SYS$QIOW(,%VAL(CHAN),
     *	    %VAL(IO$_WRITELBLK .OR. IO$M_NOFORMAT)
     *      ,,,,BUF,%VAL(80),,,,)
            CALL SYS$DASSGN(%VAL(CHAN))
#endif
	ELSE
	    IF(.NOT.TEKOPN)THEN
#ifdef unix
		OPEN(LUTK,FILE=FILET,STATUS='NEW')
#else
                OPEN(LUTK,FILE=FILET,STATUS='NEW',
     *		CARRIAGECONTROL='NONE')
#endif
		TEKOPN=.TRUE.
	    ENDIF
#if defined unix && ! defined F90_NO_FPUTC
	    do 51 i=1,80
		call fputc(LUTK,buf(i:i))
51	    continue
#else
	    WRITE(LUTK,999)BUF
999	    format(80a1)
#endif
	ENDIF
	IPT=1
	ENDIF
	IF(J.LE.ICNT)GOTO 40
	RETURN
	ENTRY CHDMP
#ifdef unix
	IF(FILET(1:8).EQ.'/dev/tty')THEN
	    do 60 i=1,ipt-1
		istat=putch(buf(i:i))
60	    continue
	ELSE
	    IF(.NOT.TEKOPN)THEN
		OPEN(LUTK,FILE=FILET,STATUS='NEW')
		TEKOPN=.TRUE.
	    ENDIF
	    do 61 i=1,ipt-1
#ifndef F90_NO_FPUTC
		call fputc(LUTK,buf(i:i))
#else
		WRITE(LUTK,'(a1)')(BUF(I:I))
#endif
61	    continue
	ENDIF
#else
        IF(FILET(1:3).EQ.'TT:')THEN
            CALL SYS$ASSIGN('TT',CHAN,,)
            CALL SYS$QIOW(,%VAL(CHAN),
     *		%VAL(IO$_WRITELBLK .OR. IO$M_NOFORMAT)
     *      ,,,,BUF,%VAL(IPT-1),,,,)
            CALL SYS$DASSGN(%VAL(CHAN))
        ELSE
            IF(.NOT.TEKOPN)THEN
                OPEN(LUTK,FILE=FILET,STATUS='NEW',
     *		CARRIAGECONTROL='NONE')
                TEKOPN=.TRUE.
            ENDIF
            WRITE(LUTK,999)(BUF(I),I=1,IPT-1)
        ENDIF
#endif
	IPT=1
	RETURN
	END
