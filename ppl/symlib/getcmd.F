	SUBROUTINE GETCMD(LINE,ILEN,COM,X,IF,M,LABEL,N,LBSTRT)
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
C**    @(#)getcmd.F	1.2    5/26/88
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
C**		CALL GETCMD(LINE,ILEN,COM,X,IF,M,LABEL,N,LBSTRT)
C**
C**	Parameters:
C**		LINE -- Line from command input file (char*255)
C**		ILEN -- Length of LINE
C**		COM -- Command value parsed  (char*30)
C**		X -- Numerical array
C**		IF -- Flag array
C**		M -- Length of numerical and flag array
C**		LABEL -- Label part of the command line  (char*255)
C**		N -- Length of LABEL
C**		LBSTRT -- Start position of LABEL
C**
C**	Description:
C**		Gets the next command from the command input, echos the
C**		input line.  Substitutes the symbols found and parses
C**		the resultant string.  Some of the system commands are
C**		also executed within this routine.  They are SET, SHOW,
C**		LIST, IF, ELSE, ENDIF, WHILE, ENDW, @.  Any other
C**		commands are parsed further and the values of X and
C**		LABEL to the calling routine.
C**
C**	History:
C**	    Written: 
C**		22-DEC-85	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-86	by Donald W. Denbo
C**		  Adds the call to PARSEX, thus enabling the parsing
C**		  of numerical values.  LIST command changed to LISTSYM.
C**		15-JUN-86	by Donald W. Denbo
C**		  Changes made to support memory buffer added
C**		 6-SEP-86	by Donald W. Denbo
C**		  Changes made to support IF ... ELSE ... ENDIF
C**		      and made to support WHILE ... ENDW
C**
C**
C     *JD* 8.2.91 Mod to support brain dead PPL -- no @ usage
        logical         tmapdebug
        common/tmdebug/ tmapdebug

#ifdef unix
	INCLUDE 'CMDCOM.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'SWITCH.INC'
	INCLUDE 'LUNITS.INC'
#else
	INCLUDE 'PPLV11INC:CMDCOM.INC'
	INCLUDE 'PPLV11INC:SYSTEM.INC'
	INCLUDE 'PPLV11INC:SWITCH.INC'
	include 'PPLV11INC:lunits.inc'
#endif
	CHARACTER COM*(*),LABEL*(*),LINE*(*),SYM*30,VALUE*255,FILE*80
	CHARACTER ICOM*255
	LOGICAL RESULT,NUM
	DIMENSION X(*),IF(*)
	INTEGER LBSTRT,WHCNT
#ifndef unix
	integer*4 status,iflag
#endif
C
C	GET NEXT COMMAND LINE
C
	DONEF=.FALSE.
1000	IF(MEMBUF)THEN
	    IF(MBUFF.GT.BUFSZE)THEN
		DONEF=.TRUE.
		RETURN
	    ENDIF
	    LINE=CMDBUF(MBUFF)
	    MBUFF=MBUFF+1
	    ILEN=LNBLK(LINE,255)
	ELSE
	    CALL GETLIN(LINE,ILEN)
	ENDIF
	IF(.NOT.TERMF.AND.LOGCMF)WRITE(LTTOUT,999)LINE(:ILEN)
999	FORMAT(1X,A)
	IF(DONEF)RETURN
	IF(ECHOF.AND..NOT.DEBUGF)CALL ECHO(LINE,ILEN)
	CALL SYMSUB(LINE,ILEN,IER,SYM,IPOS)
	IF(DEBUGF)CALL ECHO(LINE,ILEN)
	IF((IER.NE.0).AND..NOT.(SKIPIF.OR.SKIPWH))THEN
	    CALL CMDERR(IER,LINE,ILEN,IPOS)
	    GOTO 1000
	ENDIF
	CALL CMLJST(LINE,ILEN)
	CALL PARSE(LINE,ILEN,ICOM,LABEL,N,LBSTRT)
C
C	CHECK FOR ELSE AND ENDIF // IF SKIPIF IS TRUE
C
	IF(SKIPIF)THEN
	    IF(ICOM.EQ.'IF')THEN
		IFCNT=IFCNT+1
	    ELSE IF(ICOM.EQ.'ELSE'.AND.IFCNT.EQ.0)THEN
		SKIPIF=.FALSE.
	    ELSE IF(ICOM.EQ.'ENDIF')THEN
		IF(IFCNT.EQ.0)THEN
		    SKIPIF=.FALSE.
		    IFLEV=IFLEV-1
		ELSE
		    IFCNT=IFCNT-1
		ENDIF
	    ENDIF
	    GOTO 1000
	ENDIF
	IF(SKIPWH)THEN
	    IF(ICOM.EQ.'WHILE')THEN
		WHCNT = WHCNT + 1
	    ELSE IF(ICOM.EQ.'ENDW')THEN
		IF(WHCNT.EQ.0)THEN
		    SKIPWH=.FALSE.
		    WHLEV=WHLEV-1
		ELSE
		    WHCNT=WHCNT-1
		ENDIF
	    ENDIF
	    GOTO 1000
	ENDIF
C
C	CHECK FOR QUALIFIERS AND OPTIONALLY LATCH NEW DEFAULTS
C
	CALL GTQUAL(ICOM,ICLEN,IER,IPOS)
	IF(IER.NE.0)THEN
	    CALL CMDERR(IER,LINE,ILEN,IPOS)
	    GOTO 1000
	ENDIF
	IF(SLATCH)THEN
	    SLATCH=.FALSE.
	    DO 200 I=1,NFLAGS
200	    DEFLTS(I)=FLAG(I)
	ENDIF
	COM=ICOM(1:ICLEN)
C
C	TEST FOR LOCAL COMMANDS
C
	IF(COM.EQ.'SHOW')THEN
	    SYM=LABEL(:N)
	    CALL GETSYM(SYM,VALUE,NC,IER)
	    IF(IER.EQ.0)THEN
		IF(.NOT.QUIETF)WRITE(LTTOUT,997)SYM,VALUE(1:NC)
997		FORMAT(1X,A30,' = ',A)
	    ELSE
		CALL CMDERR(IER,LINE,ILEN,LBSTRT)
	    ENDIF
	ELSE IF(COM.EQ.'SET')THEN
	    CALL SETSYM(LABEL,N,IER,IST2)
	    IF(IER.NE.0)THEN
		CALL CMDERR(IER,LINE,ILEN,LBSTRT+IST2)
	    ENDIF
	ELSE IF(COM.EQ.'LISTSYM')THEN
	    ILST=0
100	    CALL LSTSYM(SYM,VALUE,NC,ILST,IER)
	    IF(IER.EQ.0)THEN
		IF(.NOT.QUIETF)WRITE(LTTOUT,986)ILST,SYM,VALUE(1:NC)
986		FORMAT(I5,2X,A30,3X,A)
		GOTO 100
	    ENDIF
	ELSE IF(COM.EQ.'DELETE')THEN
	    SYM=LABEL(:N)
	    CALL DELSYM(SYM,IER)
	    IF(IER.NE.0)THEN
		CALL CMDERR(IER,LINE,ILEN,LBSTRT+IST2)
	    ENDIF
	ELSE IF(COM(1:1).EQ.'@')THEN
C     Brain dead exclusion of @ mod
            if (.not.tmapdebug) then
	       IF(.NOT.QUIETF)WRITE(LTTOUT,'(1X,A)')
     .             'COMMAND NOT SUPPORTED IN THIS VERSION OF PPLUS: @'
               COM = ' '
               RETURN 
            end if
	    ISPC=INDEX(LINE,' ')-1
	    ICMA=INDEX(LINE,',')-1
#ifdef unix
	    ISLSH=INDEX(LINE,':')-1
#else
	    islsh=index(line,'/')-1
#endif
	    IF(ISPC.LE.0)ISPC=256
	    IF(ICMA.LE.0)ICMA=256
	    IF(ISLSH.LE.0)ISLSH=256
	    I=MIN0(ISPC,ICMA,ISLSH,ILEN)
	    FILE=LINE(2:I)
	    IF(ILEN.LT.I+2)THEN
		LABEL=' '
		I=0
	    ELSE
		I=MIN0(ISPC,ICMA,ILEN)
		LABEL=LINE(I+2:ILEN)
		I=ILEN-I-1
	    ENDIF
	    VALUE=FILE
	    ivlen=lnblk(value,255)
#ifdef unix
	    file=value
#else
	    CALL WLDFLE('SY:.PPC',VALUE(1:ivlen),FILE,STATUS,IFLAG)
#endif
	    CALL ATSTRT(FILE,LABEL,I,IER)
	ELSE IF(COM.EQ.'RETURN')THEN
	    CALL ATEND
	    IF(DONEF)RETURN
	ELSE IF(COM.EQ.'IF')THEN
	    VALUE=LINE(4:ILEN)
	    LABEL=VALUE
	    CALL UPPER(VALUE,255)
	    ITHEN=INDEX(VALUE,'THEN')
	    IF(ITHEN.LE.0)THEN
		CALL CMDERR(4,LINE,ILEN,LBSTRT)
		GOTO 1000
	    ENDIF
C
C	CALL IF PARSING ROUTINE
C
	    CALL PARSEV(LABEL,VALUE,ITHEN-1,RESULT,IER,IPOS)
	    IF(IER.NE.0)THEN
		CALL CMDERR(IER,LINE,ILEN,LBSTRT+IPOS)
		GOTO 1000
	    ENDIF
	    IFLEV=IFLEV+1
	    IFCNT=0
	    SKIPIF=.NOT.RESULT
	ELSE IF(COM.EQ.'INC')THEN
	    NN=N
	    VALUE=LABEL
	    CALL EXPEVL(VALUE,N,XX,NUM,IER)
	    IF(NUM)THEN
		XX=XX+1.0
	    ELSE
		XX=1.0
	    ENDIF
	    WRITE(VALUE,998)XX
998	    FORMAT(1PE13.5)
	    LABEL(NN+1:)=VALUE
	    CALL SETSYM(LABEL,NN+13,IER,IST2)
	ELSE IF(COM.EQ.'DEC')THEN
	    NN=N
	    VALUE=LABEL
	    CALL EXPEVL(VALUE,N,XX,NUM,IER)
	    IF(NUM)THEN
		XX=XX-1.0
	    ELSE
		XX=0.0
	    ENDIF
	    WRITE(VALUE,998)XX
	    LABEL(NN+1:)=VALUE
	    CALL SETSYM(LABEL,NN+13,IER,IST2)
	ELSE IF(COM.EQ.'ELSE')THEN
	    IF(IFLEV.GT.0)THEN
		SKIPIF=.TRUE.
	    ELSE
		CALL CMDERR(5,LINE,ILEN,1)
	    ENDIF
	ELSE IF(COM.EQ.'ENDIF')THEN
	    IF(IFLEV.GT.0)THEN
		SKIPIF=.FALSE.
		IFLEV=IFLEV-1
	    ELSE
		CALL CMDERR(5,LINE,ILEN,1)
	    ENDIF
	ELSE IF(COM.EQ.'WHILE')THEN
	    VALUE=LINE(7:ILEN)
	    LABEL=VALUE
	    CALL UPPER(VALUE,255)
	    ITHEN=INDEX(VALUE,'THEN')
	    IF(ITHEN.LE.0)THEN
		CALL CMDERR(4,LINE,ILEN,LBSTRT)
		GOTO 1000
	    ENDIF
C
C	CALL IF PARSING ROUTINE
C
	    CALL PARSEV(LABEL,VALUE,ITHEN-1,RESULT,IER,IPOS)
	    IF(IER.NE.0)THEN
		CALL CMDERR(IER,LINE,ILEN,LBSTRT+IPOS)
		GOTO 1000
	    ENDIF
	    WHLEV=WHLEV+1
	    WHCNT=0
	    SKIPWH=.NOT.RESULT
	    WHLINE(WHLEV)=LINCNT-1
	ELSE IF(COM.EQ.'ENDW')THEN
	    IF(WHLEV.LT.1)THEN
		CALL CMDERR(8,LINE,ILEN,1)
		GOTO 1000
	    ENDIF
	    REWIND(CMDLUN)
	    DO 300 I=1,WHLINE(WHLEV)
	    READ(CMDLUN,996)
996	    FORMAT(1X)
300	    CONTINUE
	    LINCNT=WHLINE(WHLEV)
	    WHLEV=WHLEV-1
	ELSE
	    CALL PARSEX(LINE,ILEN,X,IF,M,LABEL,N)
	    LABEL(N+1:)=' '
	    RETURN
	ENDIF
	GOTO 1000
	END
