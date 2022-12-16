	SUBROUTINE OPNPPL(EFILE,ELUN,DLUN,MLUN,CLUN,LTT,
     *			  KEY1,KEY2,EPLUN1,EPLUN2)
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
*	Mod to set SYM='*PPL$PLTNME' to metafile.plt by default *jd* 1.7.94
*	Mod to not reference "getlog" routine for linux port *kob* 3.11.97	
*       Mod to set SYM='*PPL$PLTNME' to value of meta_file *js* 7.25.97
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* v691 5/14 *acm* Change name MOD to TEKMOD
* v695 9/15 *acm* write values to PPL* symbols using 6 digits not 4
* 12/2022 *acm* Code cleanup: Get rid of MULTPLT, long ago replaced by viewports
*                             Clean up unused ifdefs.
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

        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'epiclun_inc.decl'
        include 'EPICLUN.INC'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'cmrdl_inc.decl'
        include 'CMRDL.INC'
        include 'plt_inc.decl'
        include 'PLT.INC'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'conlab_inc.decl'
        include 'CONLAB.INC'
        include 'labcom_inc.decl'
        include 'LABCOM.INC'
        include 'filnam_inc.decl'
        include 'FILNAM.INC'
        include 'lunits_inc.decl'
        include 'LUNITS.INC'
        include 'pen_inc.decl'
        include 'PEN.INC'
        include 'symkey_inc.decl'
        include 'SYMKEY.INC'
        include 'system_inc.decl'
        include 'SYSTEM.INC'
        include 'cmdcom_inc.decl'
        include 'CMDCOM.INC'
	include 'gkscm2.cmn'

C
C	USE OF IP FLAGS
C
C	IP(1)= BOX ON AND OFF  1=ON  0=OFF
C

	CHARACTER EFILE*(*)
C	CHARACTER EFILE*81
	INTEGER ELUN,DLUN,MLUN,CLUN,LTT,KEY1,KEY2,EPLUN1,EPLUN2
	CHARACTER TEMP*81,TERM*20,SYM*120
C
	CHARACTER TYPE*81
C
        DATA CRB,ESCB/13,27/,TEMP/'/dev/tty'/,TERM/'/dev/tty'/
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

	CALL BAUD(IR)
	CALL PLTYPE(ICODE)
C

        call getlog(type)

        BATCHF = TYPE.EQ.' '
C
        call getenv('GRAPHTERM',type)
        len=lnblk(type,81)

	IF(TYPE(1:LEN).EQ.'TEK41XX'.OR.TYPE(1:LEN).EQ.'TEK4105')THEN
		TEKMOD=4107
		CALL TKTYPE(TEKMOD)
	ELSE IF(TYPE(1:LEN).EQ.'TEK4014')THEN
		TEKMOD=4014
		CALL TKTYPE(TEKMOD)
	ENDIF
C
C	OPEN ECHO FILE
C
	IF(EFILE.NE.' ')THEN
            OPEN(ECHLUN,FILE=EFILE,STATUS='UNKNOWN')
	ENDIF
c
c	open key file
c
        call dbmopen('PPL$KEY.001',11,keyln1)
c
        IF(BATCHF)TERM='/dev/null'
C
C Don't open below -- problem w/ Ultrix *JD* 3.5.91
C       OPEN(UNIT=LTTOUT,FILE=TERM,STATUS='UNKNOWN')
C
	CALL STPRMP('ppl+>',5)
C
C	SET DEFAULT SYMBOL DEFINITIONS
C
	SYM='*PPL$FORMAT'
	ISYM=LNBLK(IFORM,80)
	CALL PUTSYM(SYM,IFORM,ISYM,IER)
	SYM='*PPL$PLTNME'
	call putsym(sym, meta_file, LNBLK(meta_file,2048), ier)
	SYM='*PPL$TEKNME'
        TEMP='/dev/tty'
	CALL PUTSYM(SYM,TEMP,7,IER)
	SYM='*PPL$XLEN'
	CALL PUTVAL(SYM,XLEN,7,IER)
	SYM='*PPL$YLEN'
	CALL PUTVAL(SYM,YLEN,7,IER)
	SYM='*PPL$XORG'
	CALL PUTVAL(SYM,XORG,7,IER)
	SYM='*PPL$YORG'
	CALL PUTVAL(SYM,YORG,7,IER)
	SYM='*PPL$WIDTH'
	CALL PUTVAL(SYM,WIDTH,7,IER)
	SYM='*PPL$HEIGHT'
	CALL PUTVAL(SYM,HEIGHT,7,IER)
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
	CALL PUTVAL(SYM,XFCT(I),7,IER)
	WRITE(SYM,999)'XOFF',I
	CALL PUTVAL(SYM,XOFF(I),7,IER)
	WRITE(SYM,999)'YFACT',I
	CALL PUTVAL(SYM,YFCT(I),7,IER)
	WRITE(SYM,999)'YOFF',I
	CALL PUTVAL(SYM,YOFF(I),7,IER)
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
10	CONTINUE
	RETURN
	END
