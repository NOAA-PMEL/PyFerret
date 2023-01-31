	SUBROUTINE SYMCNV(SYM,SYS)
C** 
C**    @(#)symcnv.f	1.2    5/26/88
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
C**		CALL SYMCNV(SYM,SYS)
C**
C**	Parameters:
C**		SYM -- Symbol name (char*120)
C**		SYS -- Symbol is SYSTEM type (logical)
C**
C**	Description:
C**		Converts the symbol name to upper case and evaluates the
C**		form SYM(#) to SYM#.
C**
C**	History:
C**	    Written:
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**

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
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
* v580 *acm* 9/04 Longer strings: changes for multi-line labels; symbols may be *2048.
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long  

	LOGICAL SYS,SYS2
	CHARACTER SYM*(*),FRMT*120,SYM2*120,VALUE*2048
	INTEGER symlen
	PARAMETER (symlen = 120)
	SYS=.FALSE.
	IF(SYM(1:1).EQ.'*')THEN
	    SYS=.TRUE.
	    FRMT=SYM(2:)
	    SYM=FRMT
	ENDIF
200	I=INDEX(SYM,'(')+1
	J=INDEX(SYM,')')-1
	CALL UPPER(SYM,symlen)
	IF(I.EQ.1.OR.J.EQ.-1)RETURN
	if(i.eq.j)then

	    read(sym(i:j),'(i1)',err=100)jk
	    frmt=sym(i:j)
	else
	WRITE(FRMT,999)J-I+1
999	FORMAT('(E',I2.2,'.0)')
	READ(SYM(I:J),FRMT,ERR=100)X
	WRITE(FRMT,998)IFIX(X+0.5)
998	FORMAT(I2.2)
	endif
	IF(FRMT(1:1).EQ.'0')THEN
	    SYM(I-1:)=FRMT(2:)
	ELSE
	    SYM(I-1:)=FRMT
	ENDIF
	RETURN
100	SYM2=SYM(I:J)
	ILEN=J-I+1
	CALL CMLJST(SYM2,ILEN)
	SYS2=SYM2(1:1).EQ.'*'
	IF(SYS2)THEN
	    FRMT=SYM2(2:)
	    SYM2=FRMT
	ENDIF
	CALL GTSYM2(SYM2,VALUE,NC,SYS2,IER)
	IF(IER.NE.0)THEN
	    SYM(I:)=' '
	    RETURN
	ELSE
	    SYM(I:)=VALUE(:NC)
	    SYM(I+NC:)=')'
	ENDIF
	GOTO 200
	END


