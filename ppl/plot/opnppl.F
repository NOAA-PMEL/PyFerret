	SUBROUTINE OPNPPL(EFILE,ELUN,DLUN,MLUN,CLUN,LTT,
     *			  KEY1,KEY2,EPLUN1,EPLUN2)
C** 
C**    @(#)opnppl.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     Mod *JD* 3.5.91 for Unix side -- do not explicitly open terminal lun
C     and use 6 rather than 5 for output -- due to buffering of output
c
c	efile  -- echo file name
c	elun   -- echo lun
c	dlun   -- data lun
c	mlun   -- mult file lun
c	clun   -- command file lun
c	ltt    -- terminal logical unit
c	key1   -- first key file lun
c	key2   -- second key file lun
c	eplun1 -- first EPIC lun
c	eplun2 -- second EPIC lun
c
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/EPICLUN.INC'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/DATA.INC'
	INCLUDE 'pplinc/CMRDL.INC'
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/LINES.INC'
	INCLUDE 'pplinc/CONLAB.INC'
	INCLUDE 'pplinc/LABCOM.INC'
	INCLUDE 'pplinc/FILNAM.INC'
	INCLUDE 'pplinc/LUNITS.INC'
	INCLUDE 'pplinc/PEN.INC'
	INCLUDE 'pplinc/SYMKEY.INC'
	INCLUDE 'pplinc/SYSTEM.INC'
	INCLUDE 'pplinc/CMDCOM.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:EPICLUN.INC'
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:DATA.INC'
        INCLUDE 'pplinc:CMRDL.INC'
        INCLUDE 'pplinc:PLT.INC'
        INCLUDE 'pplinc:LINES.INC'
        INCLUDE 'pplinc:CONLAB.INC'
        INCLUDE 'pplinc:LABCOM.INC'
        INCLUDE 'pplinc:FILNAM.INC'
        INCLUDE 'pplinc:LUNITS.INC'
        INCLUDE 'pplinc:PEN.INC'
        INCLUDE 'PPLINC:SYMKEY.INC'
        INCLUDE 'PPLINC:SYSTEM.INC'
        INCLUDE 'PPLINC:CMDCOM.INC'
#endif
C
C	USE OF IP FLAGS
C
C	IP(1)= BOX ON AND OFF  1=ON  0=OFF
C
#ifndef unix
c       Set up structures for exit handler
c
        STRUCTURE /EXIT_DESCRIPTOR/
            INTEGER LINK,
     *      ADDR,
     *      ARGS /1/,
     *      STATUS_ADDR
        END STRUCTURE
        RECORD /EXIT_DESCRIPTOR/ HANDLER
C
        EXTERNAL PPLEXT
        INTEGER*4 EXIT_STATUS
C
        BYTE BUF(80),CRB,ESCB
#endif
	CHARACTER EFILE*(*)
	INTEGER ELUN,DLUN,MLUN,CLUN,LTT,KEY1,KEY2,EPLUN1,EPLUN2
#ifndef unix
	CHARACTER ESC*1,CR*1
        INTEGER*4 LIB$GET_FOREIGN,STAT,LIB$GET_SYMBOL,LEN,IFLAG
        INCLUDE '($JPIDEF)'
#endif
	CHARACTER TEMP*81,TERM*20,SYM*30
C
	CHARACTER TYPE*81
C
#ifdef unix
	DATA CRB,ESCB/13,27/,TEMP/'/dev/tty'/,TERM/'/dev/tty'/
#else
        DATA CRB,ESCB/13,27/,TEMP/'TT:'/,TERM/'TT:'/
        EQUIVALENCE (BUF(1),LABEL),(CR,CRB),(ESC,ESCB)
C
C       SET UP THE EXIT HANDLER
C
        HANDLER.ADDR = %LOC(PPLEXT)
        HANDLER.STATUS_ADDR = %LOC(EXIT_STATUS)
C
C       EXTABLISH THE EXIT HANDLER
C
        CALL SYS$DCLEXH(HANDLER)
#endif
C
C	SET UP THE LOGICAL UNITS
C
	LUNN=DLUN
	LUN1=DLUN
	LUN11=EPLUN1
	LUN21=EPLUN2
	LMULT=MLUN
	LTTOUT=LTT
	KEYLN1=KEY1
	KEYLN2=KEY2
	CMDLUN=CLUN
	ECHLUN=ELUN
C
#ifndef unix
	CALL ERRSET(64,.TRUE.,.FALSE.,.TRUE.,.FALSE.)
	CALL ERRSET(44,.TRUE.,.FALSE.,.TRUE.,.FALSE.)
	CALL ERRSET(36,.TRUE.,.FALSE.,.TRUE.,.FALSE.)
	CALL ERRSET(30,,.FALSE.,,.FALSE.)
#endif
	CALL BAUD(IR)
	CALL PLTYPE(ICODE) 
C
#ifdef unix
	call getlog(type)
	BATCHF = TYPE.EQ.' '
C
	call getenv('GRAPHTERM',type)
	len=lnblk(type,81)
#else
        STAT=LIB$GETJPI(JPI$_MODE,,,ISTAT,TYPE,LEN)
        BATCHF = TYPE(:LEN).EQ.'BATCH'
C
        STAT=LIB$GET_SYMBOL('GRAPHTERM',TYPE,LEN)
#endif
	IF(TYPE(1:LEN).EQ.'TEK41XX'.OR.TYPE(1:LEN).EQ.'TEK4105')THEN
		MOD=4107
		CALL TKTYPE(MOD)
	ELSE IF(TYPE(1:LEN).EQ.'TEK4014')THEN
		MOD=4014
		CALL TKTYPE(MOD)
	ENDIF
C
C	OPEN ECHO FILE
C
	IF(EFILE.NE.' ')THEN
#ifdef unix
	    OPEN(ECHLUN,FILE=EFILE,STATUS='UNKNOWN')
#else
            OPEN(ECHLUN,FILE=EFILE,STATUS='NEW',CARRIAGECONTROL='LIST')
#endif
	ENDIF
c
c	open key file
c
#ifdef unix
	call dbmopen('PPL$KEY.001',11,keyln1)
c
	IF(BATCHF)TERM='/dev/null'
C
C Don't open below -- problem w/ Ultrix *JD* 3.5.91
C	OPEN(UNIT=LTTOUT,FILE=TERM,STATUS='UNKNOWN')
#else
        OPEN(KEYLN1,FILE='PPL$KEY.001',STATUS='NEW',
     *  ORGANIZATION='INDEXED',ACCESS='KEYED',
     *  RECORDTYPE='VARIABLE',FORM='FORMATTED',
     *  KEY=(1:30:CHARACTER))
c
        IF(BATCHF)TERM='SYS$OUTPUT:'
C
        OPEN(UNIT=LTTOUT,FILE=TERM,STATUS='UNKNOWN',RECL=255)
#endif
C
	CALL STPRMP('ppl>',4)
C
C	SET DEFAULT SYMBOL DEFINITIONS
C
	SYM='*PPL$FORMAT'
	ISYM=LNBLK(IFORM,80)
	CALL PUTSYM(SYM,IFORM,ISYM,IER)
	SYM='*PPL$PLTNME'
	TEMP='ppl.meta'
	CALL PUTSYM(SYM,TEMP,8,IER)
	SYM='*PPL$TEKNME'
#ifdef unix
	TEMP='/dev/tty'
#else
	TEMP='TT:'
#endif
	CALL PUTSYM(SYM,TEMP,8,IER)
	SYM='*PPL$XLEN'
	CALL PUTVAL(SYM,XLEN,4,IER)
	SYM='*PPL$YLEN'
	CALL PUTVAL(SYM,YLEN,4,IER)
	SYM='*PPL$XORG'
	CALL PUTVAL(SYM,XORG,4,IER)
	SYM='*PPL$YORG'
	CALL PUTVAL(SYM,YORG,4,IER)
	SYM='*PPL$WIDTH'
	CALL PUTVAL(SYM,WIDTH,4,IER)
	SYM='*PPL$HEIGHT'
	CALL PUTVAL(SYM,HEIGHT,4,IER)
	SYM='*PPL$LINE_COUNT'
	TEMP='0'
	CALL PUTSYM(SYM,TEMP,1,IER)
C
C	SET PPL$XFACT, XOFF, YFACT AND YOFF
C
	ISYM=MIN0(1,NLINES)
	DO 20 I=1,ISYM
	WRITE(SYM,999)'XFACT',I
999	FORMAT('*PPL$',A,'(',I3,')')
	CALL PUTVAL(SYM,XFCT(I),4,IER)
	WRITE(SYM,999)'XOFF',I
	CALL PUTVAL(SYM,XOFF(I),4,IER)
	WRITE(SYM,999)'YFACT',I
	CALL PUTVAL(SYM,YFCT(I),4,IER)
	WRITE(SYM,999)'YOFF',I
	CALL PUTVAL(SYM,YOFF(I),4,IER)
20	CONTINUE
C
C	STORE VIEW SYMBOLS
C
	WRITE(SYM,997)'X'
997	FORMAT('*PPL$VIEW_',A)
	TEMP='0.000'
	CALL PUTSYM(SYM,TEMP,5,IER)
	WRITE(SYM,997)'Y'
	CALL PUTSYM(SYM,TEMP,5,IER)
	WRITE(SYM,997)'Z'
	CALL PUTSYM(SYM,TEMP,5,IER)
10	RETURN
	END
#ifndef unix
        SUBROUTINE PPLEXT(STATUS)
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**     Subprogram called:
C**             N/A
C**
C**     Parameters:
C**             STATUS -- Exit status, passed from system
C**
C**     Description:
C**             This routine closes all files and deletes temporary files
C**
C**     History:
C**         Written: 
C**             2-AUG-86        by Donald W. Denbo
C**         Modified:
C**
C**
        INTEGER*4 STATUS
        INCLUDE 'pplinc:FILNAM.INC'
        INCLUDE 'pplinc:LUNITS.INC'
        INCLUDE 'CMDINC:SYMKEY.INC'
        INCLUDE 'CMDINC:CMDCOM.INC'
        INCLUDE 'pplinc:PLTCOM.DAT'
C
C       CLOSE ALL OPEN COMMAND FILES
C
        CALL ATFLSH
        CLOSE(KEYLN1,DISP='DELETE')
C
C       CLOSE MULT FILE IF OPEN
C
        IF(MULTFL.NE.' ')THEN
           CLOSE(LMULT,DISP='DELETE')
        ENDIF
        IF(PLTFLG)THEN
C
C       BINARY FILE IS OPEN
C
            PEN = .FALSE.
            CALL ZABMV
            CALL BINFSH
        ENDIF
        RETURN
        END      
#endif
