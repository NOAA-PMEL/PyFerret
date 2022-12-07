	SUBROUTINE ATEND
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
C**    @(#)atend.F	1.2    5/26/88
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
C**		CALL ATEND
C**
C**	Parameters:
C**
C**	Description:
C**		ATEND closes the currently open command file and user
C**		key file and returns to the previous command file.
C**
C**	History:
C**	    Written: 
C**		26-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		15-JUN-86	by Donald W. Denbo
C**		  changes for memory buffer added
C**
C*** V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*                             Remove obsolete ifdefs

	include 'SYSTEM.INC'
	include 'CMDCOM.INC'
	include 'SYMKEY.INC'
	INCLUDE 'LUNITS.INC'

	CHARACTER KFILE*30,SYM*120
	DONEF=CMDLEV.EQ.1
	IF(DONEF)THEN
C
C	DONE -- FINISH UP AND EXIT
C
	    RETURN
	ENDIF
C
C	CLOSE USER KEY FILE AND CURRENT COMMAND FILE
C
	call dbmclear(keyln2)
	keyln2=keyln2-1
	CLOSE(CMDLUN)
C
C	RESET COMMAND FILE 
C
	CMDLEV=CMDLEV-1
	IFLEV=CMIFLV(CMDLEV)
	WHLEV=CMWHLV(CMDLEV)
	SKIPIF=.FALSE.
	SKIPWH=.FALSE.
	DO I=1,WHLEV
	   WHLINE(I)=CMWHLN(CMDLEV,I)
	ENDDO
	CMDFIL=CMFILE(CMDLEV)
	LINCNT=CMLINE(CMDLEV)
	ECHOF=CMECHO(CMDLEV)
	DEBUGF=CMDEBG(CMDLEV)
	QUIETF=CMQUIE(CMDLEV)
	LOGCMF=CMLOG(CMDLEV)

	IF(CMDFIL.EQ.'/dev/tty')THEN
	    TERMF=.TRUE.
	    IF(.NOT.QUIETF)WRITE(LTTOUT,997)
997	    FORMAT(' Control returned to keyboard')
	    OPEN(CMDLUN,FILE=CMDFIL,STATUS='OLD')
	ELSE IF(CMDFIL.EQ.'$$MEMBUF$$')THEN
	    TERMF=.FALSE.
	    MEMBUF=.TRUE.
	ELSE
C
C	OPEN OLD COMMAND FILE AND SKIP TO CORRECT LINE
C
	    OPEN(CMDLUN,FILE=CMDFIL,STATUS='OLD')

998	    FORMAT(1X)
	    DO I=1,LINCNT
	       READ(CMDLUN,998)
	    ENDDO
	ENDIF
	ISYM=LNBLK(CMDFIL,80)
	SYM='*PPL$COMMAND_FILE'
	CALL PUTSYM(SYM,CMDFIL,ISYM,IER)
	IF(CMDLEV.GT.1)THEN
C
C	OPEN OLD USER KEY FILE
C
	    WRITE(KFILE,999)CMDLEV
999	    FORMAT('PPL$KEY.',I3.3)
	    call dbmopen(kfile,11,keyln2)
	ELSE
	    SMKEY2=.FALSE.
	ENDIF
	RETURN
	END
