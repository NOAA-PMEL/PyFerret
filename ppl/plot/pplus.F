	PROGRAM PPLUS
C** 
C**    @(#)pplus.F	1.2    9/8/88
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
C     Mod *JD* 3.5.91 to use unit 6 rather than 5 for unix terminal lun
C     Had problem with buffered output.  See also opnppl

#ifdef unix
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/SYSTEM.INC'
	INCLUDE 'pplinc/SWITCH.INC'
C
	CHARACTER FROM*80,TEMP*255,LINE*255,TYPE*81
	CHARACTER COMBUF(5)*255,FILE*81,DTE*24
	DATA COMBUF/5*' '/,FILE/'echo.file'/
#else
        INCLUDE 'pplinc:PLT.INC'
        INCLUDE 'PPLINC:SYSTEM.INC'
        INCLUDE 'PPLINC:SWITCH.INC'
C
        CHARACTER FROM*80,TEMP*255,LINE*255,TYPE*81
        INTEGER*4 STATUS,IFLAG,LIB$GET_SYMBOL,LEN
        CHARACTER COMBUF(5)*255,FILE*81,DTE*10,TME*8
        DATA COMBUF/5*' '/,FILE/'ECHO'/
#endif
C
C	TURN ON LOGGING AND DO PPLUS LOGGING
C
	LOGITF=.TRUE.
#ifndef unix
	CALL LOGGER('PPLUS')
#endif
C
C	INITIALIZE PLOT5+ AND SET LOGICAL UNITS
C
	ICODE=1
#ifdef unix

*     Mod *JD* 3.91 Original has unit 5 for tty  -- change to 6
C	CALL OPNPPL(FILE,23,1,16,24,5,1,2,11,21)
	CALL OPNPPL(FILE,23,1,16,24,6,1,2,11,21)
#else
        CALL OPNPPL(FILE,23,1,6,24,5,20,22,11,21)
#endif
C
C	CHECK FOR STARTUP FILE
C
#ifdef unix
	call getenv('PPL$STARTUP',type)
	len=lnblk(type,81)
	IF(len.ge.1)THEN
#else
        STATUS=LIB$GET_SYMBOL('PPL$STARTUP',TYPE,LEN)
        IF(STATUS.EQ.1)THEN
#endif
C
C	    TURN OFF ECHO AND RUN PLOT5+
C
	    ECHOF=.FALSE.
	    LINE=' '
#ifdef unix
	    from=type(1:len)
#else
	    CALL WLDFLE('SY:.PPC',TYPE(1:LEN),FROM,STATUS,IFLAG)
#endif
	    CALL PPLCMD(FROM,LINE,IS,COMBUF,5,0)
	    ECHOF=.TRUE.
	ENDIF
C
C	CHECK FOR COMMAND LINE COMMAND
C
#ifdef unix
	ic=iargc()
	if(ic.ge.1)then
	    istp=1
	    do 1 i=1,ic
		call getarg(i,temp)
		len=lnblk(temp,255)
		if(istp.eq.1)then
		    line=temp(:len)
		    istp=len+1
		else
		    line(istp:)=' '//temp(:len)
		    istp=istp+len+1
		endif
1	    continue
	else
	    line=' '
	    istp=1
	endif
	istp=istp-1
#else
        STAT=LIB$GET_FOREIGN(LINE,,ISTP)
#endif
	IS=1
#ifdef unix
	IF(INDEX(LINE(:ISTP),':Q').EQ.0)THEN
*     Mod *JD* 3.19 to write to unit 6, not 5, as in the original
C           WRITE(5,'(''  PLOT PLUS  V 1.1  05-MAR-88 (unix)'')') 
            WRITE(6,'('' PLOTPLUS  V 1.1 March 91 (unix)'')') 
	    CALL FDATE(DTE)
C *JD*      WRITE(5,9999)DTE  
	    WRITE(6,9999)DTE 
9999	    FORMAT(2x,A24) 
#else
        IF(INDEX(LINE(:ISTP),'/Q').EQ.0)THEN
            WRITE(LTTOUT,'('' PLOTPLUS  V 1.1 March 91 (vms)'')') 
            CALL DATE(DTE)
            CALL TIME(TME)
            WRITE(5,9999)DTE,TME 
9999        FORMAT(1x,A11,A15) 
#endif
	ENDIF
	IF(LINE.EQ.' '.OR.ISTP.EQ.0)THEN
C
C	NO COMMAND ON LINE
C
#ifdef unix
	    FROM='/dev/tty'
#else
	    FROM='TT:'
#endif
	    LINE=' '
	ELSE
	    TEMP=LINE(IS:ISTP)
	    ISPC=INDEX(TEMP,' ')
	    IF(ISPC.GT.0)THEN
		FROM=TEMP(:ISPC-1)
		LINE=TEMP(ISPC+1:)
		IS=ISTP-IS+1-ISPC
	    ELSE
		FROM=TEMP
		LINE=' '
		IS=0
	    ENDIF
	    TEMP=FROM
	    DEFLTS(1)=.TRUE.
	    CALL GTQUAL(TEMP,ITLEN,IER,IPOS)
	    ECHOF=SECHO
	    DEBUGF=SDEBUG
	    IF(SLATCH)THEN
		SLATCH=.FALSE.
		DO 100 I=1,NFLAGS
100		DEFLTS(I)=FLAG(I)
	    ENDIF
#ifdef unix
	    from=temp(1:itlen)
#else
	    CALL WLDFLE('SY:.PPC',TEMP(1:ITLEN),FROM,STATUS,IFLAG)
#endif
	    TERMF=.FALSE.
	    DEFLTS(1)=.FALSE.
	ENDIF

C
C	RUN PLOT5+ INTERACTIVE
C
	CALL PPLCMD(FROM,LINE,IS,COMBUF,5,0)
C
C	IF PPLCMD RETURNS THEN EXIT 
C
	CALL CLSPPL
	CALL EXIT
	END


