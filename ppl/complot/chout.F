	SUBROUTINE CHOUT(CHR,ICNT)
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
	include 'PLTCOM.DAT'
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
