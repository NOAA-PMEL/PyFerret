	SUBROUTINE ATSTRT(FILE,LINE,LEN,IER)
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
C**    @(#)atstrt.F	1.2    5/26/88
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
C**		CALL ATSTRT(FILE,LINE,LEN,IER)
C**
C**	Parameters:
C**		FILE -- File to open for next command file (char*80)
C**		LINE -- String containing symbols to be passed (char*2048)
C**		LEN  -- Length of LINE
C**		IER  -- Error code
C**			0 = success
C**			9 = command file open error
C**
C**	Description:
C**		Opens FILE as the next command file.  Also updates
C**		CMDLEV and creates the appropriate user key file.
C**
C**	History:
C**	    Written: 
C**		26-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		15-JUN-86	by Donald W. Denbo
C**		  Changes to support memory buffer added
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C v552 *acm* 3/03 Longer strings: changes for multi-line labels
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*                             Remove obsolete ifdefs

	INCLUDE 'CMDCOM.INC'
	INCLUDE 'SYMKEY.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'SWITCH.INC'
	INCLUDE 'LUNITS.INC'

	CHARACTER FILE*(*),LINE*(*),TO*80,TEMP*2048
	CHARACTER SYM*120
	INTEGER LEN
	IER=0
	CMDLEV=CMDLEV+1
	IF(CMDLEV.GT.1)THEN
	    IF(.NOT.(TERMF.OR.MEMBUF))THEN
		close(cmdlun)
		CMLINE(CMDLEV-1)=LINCNT
	    ENDIF
	    MEMBUF=.FALSE.
	    TERMF=.FALSE.
	    WRITE(TO,999)CMDLEV
999	    FORMAT('PPL$KEY.',I3.3)
	    SMKEY2=.TRUE.
	    keyln2=keyln2+1
	    call dbmopen(to,11,keyln2)
	ENDIF
	CMFILE(CMDLEV)=FILE
	CMECHO(CMDLEV-1)=ECHOF
	CMDEBG(CMDLEV-1)=DEBUGF
	CMQUIE(CMDLEV-1)=QUIETF
	CMLOG(CMDLEV-1)=LOGCMF
	CMIFLV(CMDLEV-1)=IFLEV
	CMWHLV(CMDLEV-1)=WHLEV
	DO I=1,WHLEV
	   CMWHLN(CMDLEV-1,I)=WHLINE(I)
	ENDDO
	IFLEV=0
	WHLEV=0
	DEBUGF=SDEBUG
	QUIETF=SQUIET
	LOGCMF=SLOG
	CMDFIL=FILE
	IF(CMDLEV.GT.1)ECHOF=SECHO
	LINCNT=0
	ISYM=LNBLK(FILE,80)
	SYM='*PPL$COMMAND_FILE'
	CALL PUTSYM(SYM,FILE,ISYM,IER)
	IF(.NOT.MEMBUF)THEN
	    OPEN(CMDLUN,FILE=FILE,STATUS='OLD',ERR=1000)
	ENDIF
C
C	PROCESS COMMAND LINE
C
	IL=LEN
	IF(LEN.GT.0.AND.LINE.NE.' ')THEN
C
C	    PROCESS AND PARSE COMMAND LINE
C
	    ICNT=1
100	    ISPC=INDEX(LINE,' ')
	    IF(LINE(1:1).EQ.'"')THEN
C
C		QUOTED STRING, FIND CLOSING QUOTE
C
		IQTE=0
		TEMP=LINE(2:)
		LINE=TEMP
200		ISPC=INDEX(LINE(IQTE+1:),'"')+IQTE
		IF(LINE(ISPC+1:ISPC+1).EQ.'"')THEN
		    TEMP=LINE(:ISPC)
		    TEMP(ISPC+1:)=LINE(ISPC+2:)
		    LINE=TEMP
		    IQTE=ISPC
		    IL=IL-1
		    GOTO 200
		ENDIF
	    ENDIF
	    WRITE(TO,997)ICNT
997	    FORMAT('P(',I3.3,')')
	    CALL PUTSYM(TO,LINE,ISPC-1,IER)
	    TEMP=LINE(ISPC+1:)
	    IL=IL-ISPC
	    DO I=1,IL
	       IF(TEMP(I:I).NE.' ')GOTO 11
	    ENDDO
	    RETURN
11	    LINE=TEMP(I:)
	    ICNT=ICNT+1
	    IL=IL-I+1
	    GOTO 100
	ENDIF
	RETURN
C
C	ERROR IN OPENING 'FILE' AS COMMAND INPUT
C
1000	IF(.NOT.QUIETF)WRITE(LTTOUT,996)FILE
996	FORMAT(' - Command file not found - '/1X,A79)
	CALL ATEND
	IER=9
	RETURN
	END
