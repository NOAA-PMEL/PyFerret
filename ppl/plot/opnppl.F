	SUBROUTINE OPNPPL(EFILE,ELUN,DLUN,MLUN,CLUN,LTT,
     *			  KEY1,KEY2,EPLUN1,EPLUN2)
*	Mod to set SYM='*PPL$PLTNME' to metafile.plt by default *jd* 1.7.94
*	Mod to not reference "getlog" routine for linux port *kob* 3.11.97	
*       Mod to set SYM='*PPL$PLTNME' to value of meta_file *js* 7.25.97
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
        include 'tmap_pplv11inc/parampl5_dat.decl'
        include 'pplv11inc/PARAMPL5.DAT'
        include 'tmap_pplv11inc/epiclun_inc.decl'
        include 'pplv11inc/EPICLUN.INC'
        include 'tmap_pplv11inc/axis_inc.decl'
        include 'pplv11inc/AXIS.INC'
        include 'tmap_pplv11inc/data_inc.decl'
        include 'pplv11inc/DATA.INC'
        include 'tmap_pplv11inc/cmrdl_inc.decl'
        include 'pplv11inc/CMRDL.INC'
        include 'tmap_pplv11inc/plt_inc.decl'
        include 'pplv11inc/PLT.INC'
        include 'tmap_pplv11inc/lines_inc.decl'
        include 'pplv11inc/LINES.INC'
        include 'tmap_pplv11inc/conlab_inc.decl'
        include 'pplv11inc/CONLAB.INC'
        include 'tmap_pplv11inc/labcom_inc.decl'
        include 'pplv11inc/LABCOM.INC'
        include 'tmap_pplv11inc/filnam_inc.decl'
        include 'pplv11inc/FILNAM.INC'
        include 'tmap_pplv11inc/lunits_inc.decl'
        include 'pplv11inc/LUNITS.INC'
        include 'tmap_pplv11inc/pen_inc.decl'
        include 'pplv11inc/PEN.INC'
        include 'tmap_pplv11inc/symkey_inc.decl'
        include 'pplv11inc/SYMKEY.INC'
        include 'tmap_pplv11inc/system_inc.decl'
        include 'pplv11inc/SYSTEM.INC'
        include 'tmap_pplv11inc/cmdcom_inc.decl'
        include 'pplv11inc/CMDCOM.INC'
	include 'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE 'tmap_pplv11inc:tmap_PARAMPL5.DAT'
        INCLUDE 'tmap_pplv11inc:tmap_EPICLUN.INC'
        INCLUDE 'tmap_pplv11inc:tmap_AXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_DATA.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRDL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PLT.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LINES.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CONLAB.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LABCOM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_FILNAM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LUNITS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PEN.INC'
        INCLUDE 'TMAP_PPLV11INC:SYMKEY_INC.DECL'
        INCLUDE 'PPLV11INC:SYMKEY.INC'
        INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
        INCLUDE 'PPLV11INC:SYSTEM.INC'
        INCLUDE 'TMAP_PPLV11INC:CMDCOM_INC.DECL'
        INCLUDE 'PPLV11INC:CMDCOM.INC'
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
C	CHARACTER EFILE*81
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
#ifndef F90_NO_GETLOG
        call getlog(type)
#endif
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
C       OPEN(UNIT=LTTOUT,FILE=TERM,STATUS='UNKNOWN')
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
	CALL STPRMP('ppl+>',5)
C
C	SET DEFAULT SYMBOL DEFINITIONS
C
	SYM='*PPL$FORMAT'
	ISYM=LNBLK(IFORM,80)
	CALL PUTSYM(SYM,IFORM,ISYM,IER)
	SYM='*PPL$PLTNME'
C	TEMP='metafile.plt'
C	CALL PUTSYM(SYM,TEMP,12,IER)
	call putsym(sym, meta_file, LNBLK(meta_file,128), ier)
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
#ifdef unix
        include 'tmap_pplv11inc/filnam_inc.decl'
        include 'pplv11inc/FILNAM.INC'
        include 'tmap_pplv11inc/lunits_inc.decl'
        include 'pplv11inc/LUNITS.INC'
        include 'tmap_pplv11inc/symkey_inc.decl'
        include 'pplv11inc/SYMKEY.INC'
        include 'tmap_pplv11inc/cmdcom_inc.decl'
        include 'pplv11inc/CMDCOM.INC'
        include 'tmap_pplv11inc/pltcom_dat.decl'
        include 'pplv11inc/PLTCOM.DAT'
#else
        INCLUDE 'tmap_pplv11inc:tmap_FILNAM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LUNITS.INC'
        INCLUDE 'TMAP_PPLV11INC:SYMKEY_INC.DECL'
        INCLUDE 'PPLV11INC:SYMKEY.INC'
        INCLUDE 'TMAP_PPLV11INC:CMDCOM_INC.DECL'
        INCLUDE 'PPLV11INC:CMDCOM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PLTCOM.DAT'
#endif
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
