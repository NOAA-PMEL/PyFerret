	SUBROUTINE PPLCMD(FROMI,LINEI,ISI,COMBUF,ICMDIM,ICMSZE)
C**

C**    @(#)pplcmd.F   1.4    11/8/88 !Mod added 12.6.88 JD
C**    @(#)pplcmd.F   1.3     9/8/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*       Modified 8.12.88 J Davison TMAP/JISAO/PMEL   
*       to include SHADE command.                    
*	J Davison 11.28.88 Modified for compatibility to v1.1 PPL
*	J Davison 05.04.89 Added pixmap call to capture color bitmap
*	J Davison 10.04.89 Added aline call to plot a line
*	J Davison 01.25.90 Modified CLSPLT to handle metafiles
*	J Davison 04.10.90 Mod to support 'PEN N,0' command => color index 0
*       J Davison 08.02.91 Mods to generate brain dead version of ppl
*       J Davison 11.15.91 Mod to support 'COLOR' command
*       J Davison 01.14.92 Mod to include AREA, SQFILL
*	*jd* 10.16.92 Mod to add save & restore function for pen command
*	*tc* 01.26.93 Mod to handle draftsman and spline_tension with
*	     CONSET command.  use variables 9 (spl_ten) and 10 (drman).
*	     by the way, the array size of p and iflg is 20 in cmrd.inc so
*	     a command can handle up to 20 inputs.
*	J Davison 11.09.93 Mod to add segments to advanced commands
*	J Davison 01.06.94 Mod to use PLTNME to set metafile name
*	J Davison 01.25.94 Mod to call PPLGFLUSH to flush output at each call
*	J Davison 11.19.95 Mod for 4 digit year capability
*	J Davison 11.29.95 Mod for century axis
*	S Hankin  11.25.97 Mod to add commands PLOTPOL and FILLPOL

c
c	fromi	command file or device (/dev/tty or TT: for terminal)
c	linei	input line for parameters
c	isi	length of line in characters
c	combuf() command buffer for executing commands from memory
c	icmdim	dimensioned length of combuf
c	icmsze	number of lines in combuf
c
#ifdef unix
        include 'tmap_pplv11inc/comepl_inc.decl'  ! added 9.6.88 j davison 
        include 'pplv11inc/COMEPL.INC'  ! added 9.6.88 j davison 
	include 'tmap_pplv11inc/comeps_inc.decl'
	include 'pplv11inc/COMEPS.INC'
	include 'tmap_pplv11inc/comepv_inc.decl'
	include 'pplv11inc/COMEPV.INC'
	include 'tmap_pplv11inc/parampl5_dat.decl'
	include 'pplv11inc/PARAMPL5.DAT'
	include 'tmap_pplv11inc/ppldat_inc.decl'
	include 'pplv11inc/PPLDAT.INC'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'
	include 'tmap_pplv11inc/lines_inc.decl'
	include 'pplv11inc/LINES.INC'
	include 'tmap_pplv11inc/pen_inc.decl'
	include 'pplv11inc/PEN.INC'
	include 'tmap_pplv11inc/labels_inc.decl'
	include 'pplv11inc/LABELS.INC'
	include 'tmap_pplv11inc/labcom_inc.decl'
	include 'pplv11inc/LABCOM.INC'
	include 'tmap_pplv11inc/cont_inc.decl'
	include 'pplv11inc/CONT.INC'
	include 'tmap_pplv11inc/dashzz_inc.decl'
	include 'pplv11inc/DASHZZ.INC'
	include 'tmap_pplv11inc/conlab_inc.decl'
	include 'pplv11inc/CONLAB.INC'
	include 'tmap_pplv11inc/plt_inc.decl'
	include 'pplv11inc/PLT.INC'
	include 'tmap_pplv11inc/pltl_inc.decl'
	include 'pplv11inc/PLTL.INC'
	include 'tmap_pplv11inc/hab_inc.decl'
	include 'pplv11inc/HAB.INC'
	include 'tmap_pplv11inc/data_inc.decl'
	include 'pplv11inc/DATA.INC'
	include 'tmap_pplv11inc/cmrd_inc.decl'
	include 'pplv11inc/CMRD.INC'
	include 'tmap_pplv11inc/cmrdl_inc.decl'
	include 'pplv11inc/CMRDL.INC'
	include 'tmap_pplv11inc/hd_inc.decl'
	include 'pplv11inc/HD.INC'
	include 'tmap_pplv11inc/mplot_inc.decl'
	include 'pplv11inc/MPLOT.INC'
	include 'tmap_pplv11inc/miss_inc.decl'
	include 'pplv11inc/MISS.INC'
	include 'tmap_pplv11inc/taxis_inc.decl'
	include 'pplv11inc/TAXIS.INC'
	include 'tmap_pplv11inc/tics_inc.decl'
	include 'pplv11inc/TICS.INC'
	include 'tmap_pplv11inc/tics2_inc.decl'
	include 'pplv11inc/TICS2.INC'
	include 'tmap_pplv11inc/axisl_inc.decl'
	include 'pplv11inc/AXISL.INC'
	include 'tmap_pplv11inc/xylabp_inc.decl'
	include 'pplv11inc/XYLABP.INC'
	include 'tmap_pplv11inc/view_inc.decl'
	include 'pplv11inc/VIEW.INC'
	include 'tmap_pplv11inc/zgrid_inc.decl'
	include 'pplv11inc/ZGRID.INC'
	include 'tmap_pplv11inc/filnam_inc.decl'
	include 'pplv11inc/FILNAM.INC'
	include 'tmap_pplv11inc/lunits_inc.decl'
	include 'pplv11inc/LUNITS.INC'
	include 'tmap_pplv11inc/vector_inc.decl'
	include 'pplv11inc/VECTOR.INC'
	include 'tmap_pplv11inc/system_inc.decl'
	include 'pplv11inc/SYSTEM.INC'
	include 'tmap_pplv11inc/cmdcom_inc.decl'
	include 'pplv11inc/CMDCOM.INC'
	include 'tmap_pplv11inc/symkey_inc.decl'
	include 'pplv11inc/SYMKEY.INC'
	include 'tmap_pplv11inc/switch_inc.decl'
	include 'pplv11inc/SWITCH.INC'
	include 'tmap_pplv11inc/prmcom_inc.decl'
	include 'pplv11inc/PRMCOM.INC'
	include 'tmap_pplv11inc/gkscm2.cmn'
        include 'pplv11inc/ARFILL.INC'
#else
        INCLUDE 'TMAP_PPLV11INC:tmap_comepl.inc'  ! Added 9.6.88 J Davison 
	INCLUDE 'TMAP_PPLV11INC:tmap_comeps.inc'
	INCLUDE 'TMAP_PPLV11INC:tmap_comepv.inc'
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
	INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL'
	INCLUDE 'PPLV11INC:PPLDAT.INC'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
	INCLUDE 'PPLV11INC:LINES.INC'
	INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
	INCLUDE 'TMAP_PPLV11INC:LABELS_INC.DECL'
	INCLUDE 'PPLV11INC:LABELS.INC'
	INCLUDE 'TMAP_PPLV11INC:LABCOM_INC.DECL'
	INCLUDE 'PPLV11INC:LABCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:CONT_INC.DECL'
	INCLUDE 'PPLV11INC:CONT.INC'
	INCLUDE 'TMAP_PPLV11INC:DASHZZ_INC.DECL'
	INCLUDE 'PPLV11INC:DASHZZ.INC'
	INCLUDE 'TMAP_PPLV11INC:CONLAB_INC.DECL'
	INCLUDE 'PPLV11INC:CONLAB.INC'
	INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'
	INCLUDE 'PPLV11INC:PLT.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
	INCLUDE 'PPLV11INC:PLTL.INC'
	INCLUDE 'TMAP_PPLV11INC:HAB_INC.DECL'
	INCLUDE 'PPLV11INC:HAB.INC'
	INCLUDE 'TMAP_PPLV11INC:DATA_INC.DECL'
	INCLUDE 'PPLV11INC:DATA.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRD_INC.DECL'
	INCLUDE 'PPLV11INC:CMRD.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRDL_INC.DECL'
	INCLUDE 'PPLV11INC:CMRDL.INC'
	INCLUDE 'TMAP_PPLV11INC:HD_INC.DECL'
	INCLUDE 'PPLV11INC:HD.INC'
	INCLUDE 'TMAP_PPLV11INC:MPLOT_INC.DECL'
	INCLUDE 'PPLV11INC:MPLOT.INC'
	INCLUDE 'TMAP_PPLV11INC:MISS_INC.DECL'
	INCLUDE 'PPLV11INC:MISS.INC'
	INCLUDE 'TMAP_PPLV11INC:TAXIS_INC.DECL'
	INCLUDE 'PPLV11INC:TAXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS_INC.DECL'
	INCLUDE 'PPLV11INC:TICS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS2_INC.DECL'
	INCLUDE 'PPLV11INC:TICS2.INC'
	INCLUDE 'TMAP_PPLV11INC:AXISL_INC.DECL'
	INCLUDE 'PPLV11INC:AXISL.INC'
	INCLUDE 'TMAP_PPLV11INC:XYLABP_INC.DECL'
	INCLUDE 'PPLV11INC:XYLABP.INC'
	INCLUDE 'TMAP_PPLV11INC:VIEW_INC.DECL'
	INCLUDE 'PPLV11INC:VIEW.INC'
	INCLUDE 'TMAP_PPLV11INC:ZGRID_INC.DECL'
	INCLUDE 'PPLV11INC:ZGRID.INC'
	INCLUDE 'TMAP_PPLV11INC:FILNAM_INC.DECL'
	INCLUDE 'PPLV11INC:FILNAM.INC'
	INCLUDE 'TMAP_PPLV11INC:LUNITS_INC.DECL'
	INCLUDE 'PPLV11INC:LUNITS.INC'
	INCLUDE 'TMAP_PPLV11INC:VECTOR_INC.DECL'
	INCLUDE 'PPLV11INC:VECTOR.INC'
	INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
	INCLUDE 'PPLV11INC:SYSTEM.INC'
	INCLUDE 'TMAP_PPLV11INC:CMDCOM_INC.DECL'
	INCLUDE 'PPLV11INC:CMDCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:SYMKEY_INC.DECL'
	INCLUDE 'PPLV11INC:SYMKEY.INC'
	INCLUDE 'TMAP_PPLV11INC:SWITCH_INC.DECL'
	INCLUDE 'PPLV11INC:SWITCH.INC'
	INCLUDE 'TMAP_PPLV11INC:PRMCOM_INC.DECL'
	INCLUDE 'PPLV11INC:PRMCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:GKSCM2.CMN'
        INCLUDE 'PPLV11INC:ARFILL.INC'

#endif

* special COMMON to let this routine know if FERRET is in control
#ifdef unix
        include 'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

        logical         tmapdebug
CCCCCCC Assign value below rather than initialize here.  F90 complains (linux)
CCCCCCC data	        tmapdebug/.false./

        common/tmdebug/ tmapdebug

*       Declarations added J Davison 9.2.88                    
        integer         icmdim,is,isi,icmsze,i,ier,isym        
        integer         j,k,ic,ierr,lun1,lun11                 
        integer         idum1,iy1,ix1,ix3                   
        integer         lnblk,ws_state
                                                               
        real            xm,ym,xt,xdum1,xdum2,xdum3,xdum4       
        real            xdum5,zmn,zmx,p1,x1,y1,y2,x3,x4     
        real            diff,symwid                            
                                                               
        external        lnblk,diff,symwid                      
                                                               
        logical         first_time_thru               
        data            first_time_thru/.true./                

*       End declarations 9.2.88
                                
*	Declarations added J Davison 11.30.88
	real		xft,yft,xlt,ylt,xht,yht,xtit,xtlt
	integer		nmxt,nmyt,ilen
*	End declarations 11.30.88

	integer	  ipen_save(0:nlines)
	data	  ipen_save/1,nlines*1/

	character strr*255

	CHARACTER DTE*10,TEMP*255,TEMP2*255,FILE*81,FROM*80,LINE*255
	CHARACTER SYM*30
	CHARACTER FROMI*(*),LINEI*(*)
	CHARACTER COMBUF(ICMDIM)*(*)
	CHARACTER YES*3,NO*2
	LOGICAL LEOF
	LOGICAL IFLG1,IFLG2,LABON,LABOFF
	DIMENSION X1(5),Y1(5),Y2(4),X3(2),IX3(5)
	DIMENSION X4(4)
	CHARACTER COMM(109)*7
	EQUIVALENCE (DT,X3(1)),(ITYPET,IX3(1)),(TSMX,X4(1))
	EQUIVALENCE (XLO,X1(1)),(YLO,Y1(1)),
     *		(HLAB1,Y2(1))
	INTEGER	ISTAT,IFLAG !Added back in -- deleted for some reason in v1.1
	DATA YES,NO/'YES','NO'/
	DATA FILE/' '/
	DATA COMM/'AUTO','BAUD','EXIT','PLTYPE','RWDSEQ','SIZE',
     *  'SMOOTH',
     *  'TKTYPE','XLAB','YLAB','FORMAT','CONTOUR','PLOT','READSEQ',
     *  'VARS','XAXIS','YAXIS','NLINES','LINE','LABS','LABSET',
     *  'LEVELS','CLINE','CONSET','LIST','SKPSEQ','VIEW','ENTER',
     *  'CROSS','PARAM',
     *  'SAVE','GET','XFOR','YFOR','ORIGIN','WINDOW','RWD','RD','SKP',
     *  'MULTPLT','METRIC','ENGLISH','LIMITS','PLOTV','TAXIS','RDCOM',
     *  'PLOTUV','TICS','TMIN','TMAX','TSTART','PEN','AXLEN','AXLABP',
     *  'AXLINT','AXTYPE','AXLSZE','AXNSIG','AXNMTC','AXATIC','ROTATE',
     *  'C','BOX','RLABS','AXSET',' ','LLABS','LEV','DATPT','PLTNME',
     *  'HLABS','CONPRE','CONPST','TRANSXY','MARKH','LINFIT','DFLTFNT',
     *  'GRID','TEKNME','ECHO','VELVCT','TIME','TXTYPE','TXLSZE',
     *  'TXLINT','TXNMTC','TXLABP','AUTOLAB','EVAR','DEBUG','RESET',
     *  'TITLE','VPOINT','CLSPLT','VECTOR','VECSET','VECKEY','USR',
     *  'SHADE','SHAKEY','SHASET','PIXMAP','ALINE','TMDEBUG','COLOR',
     *  'SQFILL','FILL', 'PLOTPOL', 'FILLPOL'/

!  *  'SQFILL','AREA'/ NAME CHANGE *jd*
 
	CHARACTER*15 TMPCHR

* JD 11.29.95 Mod for century axis

	LOGICAL*4 cenflag
	COMMON/CENTURY/cenflag

*	*jd* linux external stmt below inserted	
	external scale

*****************************************************************************

*       SETUP SHADE ROUTINE DEFAULTS             
        if (first_time_thru) then                
	   tmapdebug = .false.
           first_time_thru = .false.             
           call setup_defaults                   
        end if                                   

C
C	SET UP LABELT STUFF
C
	ITXTPE=ABS(LABELT)
	ITXLBP=SIGN(1,LABELT)
C
C	INITIALIZE COMMAND BUFFER
C
	FROM=FROMI
	LINE=LINEI
	IS=ISI
	MEMBUF=ICMSZE.GE.1
	BUFSZE=ICMSZE
	IF(MEMBUF)THEN
!	    M=BUFDIM No longer defined in v1.1 -- see cmdcom.inc
	    M=IBFDIM
	    M=MIN0(M,BUFSZE)
	    DO 30 I=1,M
30	    CMDBUF(I)=COMBUF(I)
	    FROM='$$MEMBUF$$'
	    LINE=' '
	    IS=0
	ENDIF
#ifdef unix
        TERMF=FROM.EQ.'/dev/tty'
#else
        TERMF=FROM.EQ.'TT:'
#endif
	CMDLEV=0
	CALL ATSTRT(FROM,LINE,IS,IER)
C
C	    ERROR OCCURRED WHILE OPENING 'FROM' AS INPUT FOR COMMANDS
C
	IF(IER.NE.0)RETURN
C
C	READ COMMAND FROM INPUT AND PARSE
C
	MBUFF=1
10	call gqops (ws_state)
!	If GKS WS is active, flush output buffer before leaving if just plotted
	if (ws_state .ge. 3) then
	  if (makep .eq. 'CONTOUR'
     .	  .or.makep .eq. 'PLOT'
     .	  .or.makep .eq. 'VIEW'
     .	  .or.makep .eq. 'PLOTV'
     .	  .or.makep .eq. 'PLOTUV'
     .	  .or.makep .eq. 'VELVCT'
     .	  .or.makep .eq. 'VECTOR'
     .	  .or.makep .eq. 'SHADE'
     .	  .or.makep .eq. 'FILL'
     .	  .or.makep .eq. 'PLOTPOL'
     .	  .or.makep .eq. 'FILLPOL'
     .	  .or. icom(1:1) .eq. '%') then
             IF ( PPL_in_FERRET ) CALL SEG_ON( )
             call pplgflush
             IF ( PPL_in_FERRET ) CALL SEG_OFF
	   endif
	endif
	makep = ' '

	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	IF (DONEF) RETURN

	DO 20 I=1,109
	    IF(ICOM.EQ.COMM(I))GOTO 21
20	CONTINUE
	IF(ICOM(1:1).EQ.'%')THEN
! 	   Mod 11.9.93 to insert advanced command output into segments
*          TMAP check that output window is active
           IF ( PPL_in_FERRET ) CALL CK_GKS_ACTIVE( *9999 ) 
*          TMAP addition to turn on segment
           IF ( PPL_in_FERRET ) CALL SEG_ON( )

	   CALL PRMTIV(X,NSIZE)
           IF ( PPL_in_FERRET ) CALL SEG_OFF
	ELSE
	    CALL SPWN
	ENDIF
	GOTO 10
21	IFLG1=(M.GE.1.AND.IFLG(1).NE.0)
	IFLG2=(M.GE.2.AND.IFLG(2).NE.0)
	LABON=(LABEL.EQ.'ON'.OR.LABEL.EQ.'on')
	LABOFF=(INDEX(LABEL,'OFF')+INDEX(LABEL,'off')).NE.0
C       Mod below for brain dead version *JD* 8.2.91
        if (tmapdebug) then
	GOTO (100,200,300,500,600,700,900,1000,1100,1200,1300,
     *	1400,1500,1600,1800,1900,2000,2100,2200,2300,2500,2600,
     *	2700,2800,2900,3300,3400,3500,3600,4000,4200,4300,
     *	4400,4500,4600,4700, 600,1600,3300,4900,5000,5100,5200,
     *	5300,5400,5500,5700,5800,5900,6000,6100,6200,6300,6400,
     *	6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,
     *	7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,
     *	8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,
     *	9800,9900,9910,9920,9930,9940,9950,9110,9120,9130,9140,
     *	9150,9160,9170,9180,9190,98,  9191,9380,9171,9172,9173),I
        else
	GOTO (100,200,300,500,99,700,900,1000,1100,1200,1300,
     *	1400,1500,99,1800,1900,2000,2100,2200,2300,2500,2600,
     *	2700,2800,2900,99,3400,3500,3600,4000,4200,4300,
     *	4400,4500,4600,4700,99,99,99,4900,5000,5100,5200,
     *	5300,5400,5500,5700,5800,5900,6000,6100,6200,6300,6400,
     *	6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,
     *	7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,
     *	8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,
     *	9800,9900,9910,9920,9930,9940,9950,9110,9120,9130,9140,
     *	9150,9160,9170,9180,9190,98,  9191,9380,9171,9172,9173),I
        end if
        goto 10

C     TMDEBUG
 98   call upper (label,255)
      if (label .eq. 'ON')  tmapdebug = .true.
      if (label .eq. 'OFF') tmapdebug = .false.
      goto 10
C
C     Brain dead response for RDing
C
 99   IF(.NOT.QUIETF)WRITE(LTTOUT,
     . '(1X,''COMMAND NOT SUPPORTED IN THIS VERSION OF PPLUS: '',A7)') 
     . ICOM(:7)
      GOTO 10
C     AUTO
100	IF(LABON)THEN
	    IPAUSE=0
	    IHDCPY=1
	ELSE
	    IPAUSE=1
	    IHDCPY=0
	END IF
	GOTO 10
C     BAUD
200	IR=IFIX(P(1))
	CALL BAUD(IR)
	GOTO 10
C     EXIT
300	CALL ATFLSH
	RETURN
C     PLTYPE
500	ICODE=IFIX(P(1))
C       More brain dead mods:
        IF (ICODE .NE. 3 .AND. .not. tmapdebug) THEN
           IF(.NOT.QUIETF)WRITE(LTTOUT,'(1X,A)')
     .          'PLTYPE 3 ONLY IS SUPPORTED IN THIS VERSION OF PPLUS'
           GOTO 10
        ENDIF
	CALL PLTYPE(ICODE)
	CALL TKTYPE(MOD)
	GOTO 10
C     SIZE
700	WIDTH=P(1)
	HEIGHT=P(2)
	SYM='*PPL$WIDTH'
	CALL PUTVAL(SYM,WIDTH,4,IER)
	SYM='*PPL$HEIGHT'
	CALL PUTVAL(SYM,HEIGHT,4,IER)
	GOTO 10
C     SMOOTH
900	IF(M.EQ.0)GOTO 10
	CALL SMOOTH(X,NX,NY,NX,NY,IFIX(P(1)))
	GOTO 10
C     TKTYPE
1000	MOD=IFIX(P(1))
	CALL TKTYPE(MOD)
	GOTO 10
C     XLAB
1100	XLAB=LABEL
	GOTO 10
C     YLAB
1200	YLAB=LABEL
	GOTO 10
C     FORMAT
1300	IFORM=LABEL
	CALL UPPER(IFORM,80)
	SYM='*PPL$FORMAT'
	ISYM=LNBLK(IFORM,80)
	CALL PUTSYM(SYM,IFORM,ISYM,IER)
	LSCTD=SCTD
	GOTO 10
C     VARS
1800	IEFLG=0
	LSOFFS=SOFFS
	LSTIME=STIME
	LSNEXT=SNEXT
	IF(IFLG(1).NE.0)NGRP=IFIX(P(1))
	JX=0
	JY=0
	JZ=0
	DO 1801 I=2,M
	    IF(IFLG(I).EQ.0)GOTO 1801
	    IF(P(I).EQ.1.)JX=I-1
	    IF(P(I).EQ.2.)JY=I-1
	    IF(P(I).EQ.3.)JZ=I-1
1801	CONTINUE
	NVAR=M-1
	GOTO 10
C     XAXIS
1900	DO 1901 I=1,M
	    IF(I.GE.4)GOTO 1902
	    IF(IFLG(I).NE.0)X1(I)=P(I)
1901	CONTINUE
1902	IF(M.GT.0)THEN
	    IAUTOX=0
	    X1(3)=SIGN(ABS(X1(3)),X1(2)-X1(1))
	ELSE
	    IAUTOX=1
	ENDIF
	GOTO 10
C     YAXIS
2000	DO 2001 I=1,M
	    IF(M.GE.4)GOTO 2002
	    IF(IFLG(I).NE.0)Y1(I)=P(I)
2001	CONTINUE
2002	IF(M.GT.0)THEN
	    IAUTOY=0
	    Y1(3)=SIGN(ABS(Y1(3)),Y1(2)-Y1(1))
	ELSE
	    IAUTOY=1
	ENDIF
	GOTO 10
C     NLINES
2100	CALL RSMNMX
	LINEN=0
	IBASE=1
	PBUF=0
	LNUM=1
	GOTO 10
2200	I=IFIX(P(1))
	IF(IFLG2)IMARK(I)=IFIX(P(2))
	IF(M.GE.3.AND.IFLG(3).NE.0)ITYPE(I)=IFIX(P(3))
        if(m.ge.4.and.iflg(4).ne.0)xoff(i)=p(4)     
        if(m.ge.5.and.iflg(5).ne.0)yoff(i)=p(5)     
        if(m.ge.6.and.iflg(6).ne.0)dn1(i)=p(6)      
        if(m.ge.7.and.iflg(7).ne.0)up1(i)=p(7)      
        if(m.ge.8.and.iflg(8).ne.0)dn2(i)=p(8)      
        if(m.ge.9.and.iflg(9).ne.0)up2(i)=p(9)      
	IF(M.GE.10.AND.IFLG(10).NE.0)IPEN(I)=IFIX(P(10))
	GOTO 10

!	All references to X2 deleted for 1.4 version of PPLCMD
!	N=M
!	IF(M.GE.10)N=9
!	DO 2201 J=4,N
!	    IF(J.LT.6)THEN
!		K=J-3
!	    ELSE
!		K=J-1
!	    ENDIF
!	    IF(IFLG(J).NE.0)X2(I,K)=P(J)
!2201	CONTINUE

C     LABS
2300	IF(P(1).EQ.0..OR.M.EQ.0)THEN
	    DO 2301 I=1,NLABS
		LLTYPE(I)=0
2301	    RHLABS(I)=-1.0
	    NLAB=0
	ELSE
	    I=IFIX(P(1))
	    IF(N.NE.0)LABS(I)=LABEL
	    USRLBS(I)=SUSER
	    IF(IFLG2)THEN
		XLABS(I)=P(2)
		YLABS(I)=P(3)
		IF(M.GE.4.AND.IFLG(4).NE.0)LTYPE(I)=P(4)
	    ELSE
		XLABS(I)=1.E35
	    ENDIF
	    IF(NLAB.LT.I)NLAB=I
	END IF
	GOTO 10
C     LABSET
2500	DO 2501 I=1,M
	    IF(IFLG(I).NE.0)Y2(I)=P(I)
2501	CONTINUE
	GOTO 10
C     LEVELS
2600	CALL LEVEL
	GOTO 10
C     CLINE
2700	DO 2701 I=1,M
	    IF(IFLG(I).NE.0)LWGT(I)=IFIX(P(I))
2701	CONTINUE
	DO 2702 I=M+1,NLEV
2702	LWGT(I)=1
	GOTO 10
C     CONSET
2800	IF(IFLG1)HGT=P(1)
	IF(IFLG2)NSIGC=IFIX(P(2))
	IF(M.GE.3.AND.IFLG(3).NE.0)NARC=IFIX(P(3))
	IF(M.GE.4.AND.IFLG(4).NE.0)DASHLN=P(4)
	IF(M.GE.5.AND.IFLG(5).NE.0)SPACLN=P(5)
	IF(M.GE.6.AND.IFLG(6).NE.0)CAY=P(6)
	IF(M.GE.7.AND.IFLG(7).NE.0)NRNG=IFIX(P(7))
	IF(M.GE.8.AND.IFLG(8).NE.0)DSLAB=P(8)
! TMAP mod, spline fit:
	if(m.ge.9.and.iflg(9).ne.0)spline_tension=p(9)
! TMAP mod, horz contour labels:
	if(m.ge.10.and.iflg(10).ne.0.and.p(10).ne.0.)draftsman=.true.	
	if(m.ge.10.and.iflg(10).ne.0.and.p(10).eq.0.)draftsman=.false.	
	GOTO 10
C     CROSS
3600	ICROSS=0
	IF(LABON)ICROSS=1
	IF(IFLG(1).EQ.1)ICROSS=ICROSS+INT(P(1))
	GOTO 10
C     PARAM
4000	GOTO 10
C	DO 4001 I=1,M
C4001	IF(IFLG(I).EQ.1)A(I)=P(I)
C	GOTO 10
C     SAVE
4200	IF(N.EQ.0)GOTO 10
#ifdef unix
        temp=label
#else
	CALL WLDFLE('SY:.PPL_SAVE',LABEL,TEMP,ISTAT,IFLAG)
#endif
	OPEN(UNIT=LMULT,FILE=TEMP,STATUS='NEW',
     *	FORM='UNFORMATTED')
C
C	SAVE THE FOLLOWING COMMON BLOCKS:
C
C	HD
C	MPLOT   (IMULT)
C	PPLDAT  (IBASE AND PBUF)
C	SWITCH  (DEFLTS AND FLAG)
C	SYSTEM
C	XYLABP
C
	WRITE(LMULT)IPTS,IMULT,IBASE,PBUF,DEFLTS,FLAG,ECHOF,DEBUGF,
     *	BATCHF,ERRORF,DONEF,MEMBUF,LOGITF,QUIETF,LOGCMF,XLABS,YLABS,
     *	RLABS,LTYPE,XLLABS,YLLABS,LLTYPE,RHLABS,USRLBS
C
	CALL SAVE(LMULT)
	CLOSE(LMULT)
	GOTO 10
C	GET
4300	IF(N.EQ.0)GOTO 10
#ifdef unix
        temp=label
#else
        CALL WLDFLE('SY:.PPL_SAVE',LABEL,TEMP,ISTAT,IFLAG)
#endif
	CALL WLDFLE('SY:.PPL_SAVE',LABEL,TEMP,ISTAT,IFLAG)
	OPEN(UNIT=LMULT,FILE=TEMP,STATUS='OLD',
     *	FORM='UNFORMATTED',ERR=4301)
	IF(IMULT.EQ.1)THEN
	    XM=HEIGHT
	    YM=WIDTH
	ENDIF
C
C	GET THE FOLLOWING COMMON BLOCKS:
C
C	HD
C	MPLOT   (IMULT)
C	PPLDAT  (IBASE AND PBUF)
C	SWITCH  (DEFLTS AND FLAG)
C	SYSTEM
C	XYLABP
C
	READ(LMULT)IPTS,IMULT,IBASE,PBUF,DEFLTS,FLAG,ECHOF,DEBUGF,
     *	BATCHF,ERRORF,DONEF,MEMBUF,LOGITF,QUIETF,LOGCMF,XLABS,YLABS,
     *	RLABS,LTYPE,XLLABS,YLLABS,LLTYPE,RHLABS,USRLBS
C
	CALL GET(LMULT)
	CLOSE(LMULT)
	IF(IMULT.EQ.1)THEN
	    HEIGHT=XM
	    WIDTH=YM
	ENDIF
	CALL BAUD(IR)
	CALL PLTYPE(ICODE)
	CALL TKTYPE(MOD)
	GOTO 10
4301	IF(.NOT.QUIETF)WRITE(LTTOUT,'(1X,A30,'' NOT FOUND'')')LABEL
	GOTO 10
C     XFOR
4400	IFRX=LABEL
	GOTO 10
C     YFOR
4500	IFRY=LABEL
	GOTO 10
C     ORIGIN
4600	IF(IFLG1)XORG=P(1)
	IF(IFLG2)YORG=P(2)
	SYM='*PPL$XORG'
	CALL PUTVAL(SYM,XORG,4,IER)
	SYM='*PPL$YORG'
	CALL PUTVAL(SYM,YORG,4,IER)
	if(pltopn)then
	    call setax(x,nsize,xft,yft,xlt,ylt,xht,yht,nmxt,nmyt,xtit,xtlt)
     	    call scale(xft,yft,xorg,yorg,xlo,ylo)
	endif
	GOTO 10
C     WINDOW
4700	IWIND=0
	IF(LABON) IWIND=1
	GOTO 10
C     METRIC
5000	CALL METRIC
	GOTO 10
C     ENGLISH
5100	CALL ENGLSH
	GOTO 10
C     LIMITS
5200	CALL UPPER(LABEL,255)
	IF(INDEX(LABEL,'XLE').NE.0)THEN
	    XLE=.NOT.LABOFF
	    IF(IFLG1)CMXLE=P(1)
	ELSE IF(INDEX(LABEL,'XEQ').NE.0)THEN
	    XEQ=.NOT.LABOFF
	    IF(IFLG1)CMXEQ=P(1)
	ELSE IF(INDEX(LABEL,'XGE').NE.0)THEN
	    XGE=.NOT.LABOFF
	    IF(IFLG1)CMXGE=P(1)
	ELSE IF(INDEX(LABEL,'YLE').NE.0)THEN
	    YLE=.NOT.LABOFF
	    IF(IFLG1)CMYLE=P(1)
	ELSE IF(INDEX(LABEL,'YEQ').NE.0)THEN
	    YEQ=.NOT.LABOFF
	    IF(IFLG1)CMYEQ=P(1)
	ELSE IF(INDEX(LABEL,'YGE').NE.0)THEN
	    YGE=.NOT.LABOFF
	    IF(IFLG1)CMYGE=P(1)
	ELSE IF(INDEX(LABEL,'ZLE').NE.0)THEN
	    ZLE=.NOT.LABOFF
	    IF(IFLG1)CMZLE=P(1)
	ELSE IF(INDEX(LABEL,'ZEQ').NE.0)THEN
	    ZEQ=.NOT.LABOFF
	    IF(IFLG1)CMZEQ=P(1)
	ELSE IF(INDEX(LABEL,'ZGE').NE.0)THEN
	    ZGE=.NOT.LABOFF
	    IF(IFLG1)CMZGE=P(1)
	ENDIF
	GOTO 10
C     TAXIS
5400	ITFLG=0
	IF(LABON)THEN
	    IF(SYAXIS)THEN
		ITFLG=-1
	ELSE
		ITFLG=1
	    ENDIF
	ENDIF
	DO 5401 I=1,M
	    IF(I.GE.3)GOTO 5402
	    IF(IFLG(I).NE.0)X3(I)=P(I)
5401	CONTINUE
	GOTO 10
5402	DO 5403 J=I,M
	    IF(IFLG(J).NE.0)IX3(J-2)=IFIX(P(J))
5403	CONTINUE
	GOTO 10
C     RDCOM
5500	goto 10
C     TICS
5800	DO 5801 I=1,M
	    IF(I.GT.4)GOTO 5802
	    IF(IFLG(I).NE.0)X4(I)=P(I)
5801	CONTINUE
5802	IF(M.GE.5.AND.IFLG(5).EQ.1)ITX=P(5)
	IF(M.GE.6.AND.IFLG(6).EQ.1)ITY=P(6)
	GOTO 10
C	TMIN
5900	IF(N.NE.0)THEN
	    IT1=LABEL(2:11)
	    IF(ITSTRT.EQ.' ')ITSTRT=IT1
	ENDIF

C	Linux port. *jd* 12.96 F90 compiler has problem with the statement
C	below "Invalid forward reference to 6102 detected at DT (end of stmt)"
C	This is into an IF THEN ELSE clause.  As there is but one command 
C	before another GOTO I have copied the code from line 6102 and 
C	GOTO 10 right here.

CCC	GOTO 6102
	TLO=1.-DIFF(IT1,ITSTRT)/DT ! Needs to be == line 6102
	GOTO 10
C	TMAX
6000	IF(N.NE.0)THEN
	    IT2=LABEL(2:11)
	ENDIF
	GOTO 10
C	TSTART
6100	IF (IFORM(1:3).EQ.'EPI' .OR. IFORM(1:3).EQ.'BIB') THEN
	    GOTO 10
	ELSE
	    IF(N.EQ.0)LABEL(2:)=IT1
	    ITSTRT=LABEL(2:11)
6102	    TLO=1.-DIFF(IT1,ITSTRT)/DT
	    GOTO 10
	ENDIF
C	PEN
6200	I=IFIX(P(1))
	if (label .ne. ' ') then 
	   call upnsquish (label,strr,end)
	   if (strr .eq. 'SAVE') then
	      ipen_save(i) = ipen(i)
	      goto 6201
	   elseif (strr .eq. 'RESTORE') then
	      ipen(i) = ipen_save(i)
	      goto 6201
	   endif
	endif
	IF(M.LT.2.OR.IFLG(2).EQ.0)GOTO 10
	IF(I.LT.1)I=0
	IPEN(I)=IFIX(P(2))
	if (ipen(i) .eq. 0) ipen(i) = 19  !JD 10.16.92 [post atc] for 'PEN N,0'
6201	GOTO 10
C	AXLEN
6300	IF(IFLG1)XLEN=P(1)
	IF(IFLG2)YLEN=P(2)
	SYM='*PPL$XLEN'
	CALL PUTVAL(SYM,XLEN,4,IER)
	SYM='*PPL$YLEN'
	CALL PUTVAL(SYM,YLEN,4,IER)
	if(pltopn)then
	    call setax(x,nsize,xft,yft,xlt,ylt,xht,yht,nmxt,
     *	    nmyt,xtit,xtlt)
	    call scale(xft,yft,xorg,yorg,xlo,ylo)
	endif
	GOTO 10
C	AXLABP
6400	IF(IFLG1)LABELX=IFIX(P(1))
	IF(IFLG2)LABELY=IFIX(P(2))
	GOTO 10
C	AXLINT
6500	IF(IFLG1)LINTX=IFIX(P(1))
	IF(IFLG2)LINTY=IFIX(P(2))
	GOTO 10
C	AXTYPE
6600	IF(IFLG1)ITYPEX=IFIX(P(1))
	IF(IFLG2)ITYPEY=IFIX(P(2))
	GOTO 10
C	AXLSZE
6700	IF(IFLG1)XCSIZE=P(1)
	IF(IFLG2)YCSIZE=P(2)
	GOTO 10
C	AXNSIG
6800	IF(IFLG1)NSIGX=IFIX(P(1))
	IF(IFLG2)NSIGY=IFIX(P(2))
	GOTO 10
C	AXNMTC
6900	IF(IFLG1)NMTCX=IFIX(P(1))
	IF(IFLG2)NMTCY=IFIX(P(2))
	GOTO 10
C	AXATIC
7000	IF(IFLG1)NTICX=IFIX(P(1))
	IF(IFLG2)NTICY=IFIX(P(2))
	GOTO 10
C	ROTATE
7100	IF(LABON)THEN
	    CALL ROTATE(90.)
	ELSE
	    CALL ROTATE(0.)
	ENDIF
	GOTO 10
C	COMMENT COMMAND 'C'
7200	GOTO 10
C	BOX
7300	IP(1)=0
	IF(LABON)IP(1)=1
	GOTO 10
C	RLABS
7400	I=IFIX(P(1))
	IF(IFLG2)THEN
	    RLABS(I)=P(2)
	ELSE
	    RLABS(I)=0.
	ENDIF
	GO TO 10
C	AXSET
7500	DO 7501 I=1,M
7501	    IF(IFLG(I).NE.0) IAXON(I)=P(I)
	    GOTO 10
C	'BLANK COMMAND'
7600	    CONTINUE
	    GOTO 10
C	LLABS
7700	    IF(.NOT.IFLG1)GOTO 10
	    I=P(1)
	    IF(IFLG2)THEN
		XLLABS(I)=P(2)
		YLLABS(I)=P(3)
	    ELSE
		LLTYPE(I)=0
	    ENDIF
	    IF(M.GE.4.AND.IFLG(4).NE.0)LLTYPE(I)=P(4)
	    GOTO 10
C	LEV
7800	    CALL LEV
	    GOTO 10
C	DATPT
7900	    IF(IFLG1)IZTYP=P(1)
	    IF(IFLG2)IZMRK=P(2)
	    GOTO 10
C	PLTNME
*8000	    IF(N.EQ.0)THEN
*		TEMP='ppl.meta'
*	    ELSE
*		TEMP=LABEL
*	    ENDIF
*	    CALL PLTNME(TEMP)
*	    SYM='*PPL$PLTNME'
*	    ISYM=LNBLK(TEMP,255)
*	    CALL PUTSYM(SYM,TEMP,ISYM,IER)
8000	    IF(N.EQ.0)THEN
		meta_file='metafile.plt'
		TEMP='metafile.plt'
	    ELSE
		meta_file=LABEL
		TEMP=LABEL
	    ENDIF
	    SYM='*PPL$PLTNME'
	    ISYM=LNBLK(TEMP,255)
	    CALL PUTSYM(SYM,TEMP,ISYM,IER)
	    GOTO 10
C	HLABS
8100	    IF(.NOT.IFLG1)GOTO 10
	    I=P(1)
	    IF(IFLG2)THEN
		RHLABS(I)=P(2)
	    ELSE
		RHLABS(I)=-1.0
	    ENDIF
	    GOTO 10
C	CONPRE
8200	    IF(N.EQ.0)THEN
		CONPRE=' '
	    ELSE
		CONPRE=LABEL
	    ENDIF
	    GOTO 10
C	CONPST
8300	    IF(N.EQ.0)THEN
		CONPST=' '
	    ELSE
		CONPST=LABEL
	    ENDIF
	    GOTO 10
C	TRANSXY
8400	    I=IFIX(P(1))
	    IF(IFLG2)XFCT(I)=P(2)
	    IF(M.GE.3.AND.IFLG(3).NE.0)XOFF(I)=P(3)
	    IF(M.GE.4.AND.IFLG(4).NE.0)YFCT(I)=P(4)
	    IF(M.GE.5.AND.IFLG(5).NE.0)YOFF(I)=P(5)
	    IF(M.EQ.1)THEN
		XFCT(I)=1.0
		XOFF(I)=0.0
		YFCT(I)=1.0
		YOFF(I)=0.0
	    ENDIF
	    WRITE(SYM,997)'XFACT',I
997	    FORMAT('*PPL$',A,'(',I3,')')
	    CALL PUTVAL(SYM,XFCT(I),4,IER)
	    WRITE(SYM,997)'XOFF',I
	    CALL PUTVAL(SYM,XOFF(I),4,IER)
	    WRITE(SYM,997)'YFACT',I
	    CALL PUTVAL(SYM,YFCT(I),4,IER)
	    WRITE(SYM,997)'YOFF',I
	    CALL PUTVAL(SYM,YOFF(I),4,IER)
	    GOTO 10
C	MARKH
8500	    I=IFIX(P(1))
	    IF(IFLG2)HMARK(I)=P(2)
	    IF(M.EQ.1)HMARK(I)=0.08
	    GOTO 10
C	LINFIT
8600	    I=IFIX(P(1))
	    IF(I.GT.LINEN.OR.LLENG(I).LE.2.OR.LINEN.GE.NLINES)THEN
		IF(.NOT.QUIETF)WRITE(LTTOUT,
     *		'('' LINFIT ERROR -- NO ROOM OR LINE'',I3,
     *		'' DOES NOT EXIST'')')
		GOTO 10
	    ENDIF
	    CALL LINFIT(I,X,NSIZE,IBASE,PBUF)
	    GOTO 10
C	DFLTFNT
8700	    IF(N.EQ.0)GOTO 10
	    I=INDEX(LABEL,'@')
	    DTE='@CL@'
	    IF(I.EQ.0)THEN
		DTE(5:)=LABEL
	    ELSE
		DTE(4:)=LABEL
	    ENDIF
	    XT=SYMWID(HLAB1,6,DTE)
	    CONPRE=DTE(4:)
	    GOTO 10
C	GRID
8800	    IF(N.EQ.0)CAY=ABS(CAY)
	    CALL UPPER(LABEL,255)
	    IF(LABEL(1:1).EQ.'L')CAY=-ABS(CAY)
	    GOTO 10
C	TEKNME
8900	    IF(N.EQ.0)THEN
#ifdef unix
                TEMP='/dev/tty'
#else
		TEMP='TT:'
#endif
	    ELSE
		TEMP=LABEL
	    ENDIF
	    CALL TEKNME(TEMP)
	    SYM='*PPL$TEKNME'
	    ISYM=LNBLK(TEMP,255)
	    CALL PUTSYM(SYM,TEMP,ISYM,IER)
	    GOTO 10
C	ECHO
9000	    ECHOF=LABON
	    GOTO 10
C	TIME
9200	    IF(N.EQ.0)THEN
		IAUTOT=1
		GOTO 10
	    ENDIF
	    IAUTOT=0
C
C	GET TMIN,TMAX,TSTART FROM LABEL
C
C	IF NO FIRST PARAMETER SKIP TMIN PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9201
	    IC=INDEX(LABEL,',')
	    IF(IC.EQ.0)IC=256
	    IS=INDEX(LABEL,' ')
	    IS=MIN0(IS,IC)

!*JD*	    IT1=LABEL(2:IS-1)
	    TMPCHR = LABEL(1:IS-1)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		IT1=TMPCHR(2:11)         ! old format
	 	IT1(11:14) = '0019'
	    ELSE
		IT1(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (tmpchr(11:12) .EQ. ' ') IT1(11:12)='00' 
		IT1(13:14) = TMPCHR(2:3) ! century
	    ENDIF

	    IF(ITSTRT.EQ.' ')ITSTRT=IT1
	    IF(IC.NE.0)IS=IS+1
9210	    IF(LABEL(IS:IS).EQ.' ')THEN
		IS=IS+1
		IF(IS.GT.255)GOTO 9203
		GOTO 9210
	    ELSE IF(LABEL(IS:IS).EQ.',')THEN
		LABEL=LABEL(IS+1:)
		GOTO 9202
	    ELSE
		LABEL=LABEL(IS:)
	    ENDIF
C
C	IF NO SECOND PARAMETER SKIP TMAX PROCESSING
C
9201	    IF(M.GE.2.AND.IFLG(2).EQ.0)GOTO 9202
	    IC=INDEX(LABEL,',')
	    IF(IC.EQ.0)IC=256
	    IS=INDEX(LABEL,' ')
	    IS=MIN0(IS,IC)

!*JD*	    IT2=LABEL(2:IS-1)
	    TMPCHR = LABEL(1:IS-1)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		IT2=TMPCHR(2:11)         ! old format
		IT2(11:14) = '0019'
	    ELSE
		IT2(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (TMPCHR(11:12) .EQ. ' ') IT2(11:12)='00' 
		IT2(13:14) = TMPCHR(2:3) ! century
	    ENDIF

	    IF(IC.NE.0)IS=IS+1
9220	    IF(LABEL(IS:IS).EQ.' ')THEN
		IS=IS+1
		IF(IS.GT.255)GOTO 9203
		GOTO 9220
	    ELSE
		LABEL=LABEL(IS:)
	    ENDIF

!*JD* 9202  ITSTRT=LABEL(2:11)
9202	    TMPCHR = LABEL(1:15)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		ITSTRT=TMPCHR(2:11)         ! old format
		ITSTRT(11:14) = '0019'
	    ELSE
		ITSTRT(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (TMPCHR(11:12) .EQ. ' ') ITSTRT(11:12)='00' 
		ITSTRT(13:14) = TMPCHR(2:3) ! century
	    ENDIF

9203	    TLO=1.-DIFF(IT1,ITSTRT)/DT
	    GOTO 10
C	TXTYPE
9300	    CALL UPPER(LABEL,255)
C
C	IF NO FIRST PARAMETER SKIP PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9301
	    IF(LABEL(1:1).EQ.'D')THEN
		ITYPET=0
	    ELSE IF(LABEL(1:1).EQ.'M')THEN
		ITYPET=1
	    ELSE
		ITYPET=3
	    ENDIF
C
C	PROCESS TIME AXIS STYLE
C
9301	    IF(ITYPET.EQ.0)THEN
		IF(INDEX(LABEL,'HRD').NE.0)THEN
		    ITXTPE=2
		ELSE
		    ITXTPE=1
		ENDIF
	    ELSE IF(ITYPET.EQ.1)THEN
		IF(INDEX(LABEL,'DAYM').NE.0)THEN
		    ITXTPE=2
		ELSE
		    ITXTPE=1
		ENDIF
	    ELSE
		IF(INDEX(LABEL,'MON1').NE.0)THEN
		    ITXTPE=3
		ELSE IF(INDEX(LABEL,'MONY').NE.0)THEN
		    ITXTPE=2
		ELSE
		    ITXTPE=1
		ENDIF

*               Century axis mod
	        IF(INDEX(LABEL,'MULTIY').NE.0) THEN
		  CENFLAG = .TRUE.
	        ELSE
	          CENFLAG = .FALSE.		   
	        ENDIF
	    ENDIF
	    LABELT=ITXTPE*ITXLBP
	    GOTO 10
C	TXLSZE
9400	    IF(IFLG1)TCSIZE=P(1)
	    GOTO 10
C	TXLINT
9500	    IF(IFLG1)IFDB=P(1)
	    IF(IFLG2)ISDB=P(2)
	    GOTO 10
C	TXNMTC
9600	    IF(IFLG1)NMTCT=P(1)
	    GOTO 10
C	TXLABP
9700	    IF(IFLG1)THEN
		ITXLBP=P(1)
		LABELT=ITXTPE*ITXLBP
	    ENDIF
	    GOTO 10
C	AUTOLAB
9800	    IGTLAB=0
	    IF(LABON)IGTLAB=1
	    GOTO 10
C	EVAR
9900	    CALL UPPER(LABEL,255)
	    LSOFFS=SOFFS
	    LSTIME=STIME
	    LSNEXT=SNEXT
	    CALL EPICVAR
	    GOTO 10
C	DEBUG
9910	    DEBUGF=LABON
	    GOTO 10
C	RESET
9920	    LABEL='PPL$RESET'
	    N=9
	    GOTO 4300
C	TITLE
9930	    IF(N.EQ.0)THEN
		LAB1=' '
	    ELSE
		LAB1=LABEL
	    ENDIF
	    IF(IFLG1)HLAB1=P(1)
	    GOTO 10
C	VPOINT
9940	    IF(IFLG1)VIEWX=P(1)
	    IF(IFLG2)VIEWY=P(2)
	    IF(M.GE.3.AND.IFLG(3).NE.0)VIEWZ=P(3)
	    WRITE(SYM,999)'X'
999	    FORMAT('*PPL$VIEW_',A1)
	    CALL PUTVAL(SYM,VIEWX,4,IER)
	    WRITE(SYM,999)'Y'
	    CALL PUTVAL(SYM,VIEWY,4,IER)
	    WRITE(SYM,999)'Z'
	    CALL PUTVAL(SYM,VIEWZ,4,IER)
	    GOTO 10
C	CLSPLT
9950	    CALL PLTYPE(ICODE)
*	    MOD *JD* 1.25.90 to accomodate use of clsplt and metafiles
	    if (meta_open .and. .not. meta_clsplt .and. .not. first_meta) then
	       if (meta_actv) call gdawk (meta_wsid)
	       call gclwk (meta_wsid)
	       close (meta_lun)
	       meta_clsplt = .true.
	       vpt_was_set = .false.
	    end if        
	    GOTO 10
C     RWDSEQ & RWD
600	    CALL OPENF(FILE,IERR)
	    IF(IERR.NE.0)GOTO 10
	    SYM='*PPL$EOF'
	    CALL PUTSYM(SYM,NO,2,IER)
	    IF(IFORM(1:3).EQ.'DSF'.OR.IFORM(1:4).EQ.'BIBO'
     *	    .OR.IFORM(1:3).EQ.'PPL')THEN
	    CALL RWDDSF(FILE)
	    GOTO 10
	ELSE IF(IFORM(1:3).EQ.'EPI')THEN
	    rewind (LUN11)
	    GOTO 10
	ENDIF
	REWIND LUNN
	GOTO 10
C     CONTOUR
1400	MAKEP='CONTOUR'
	if(iflg1)then
	    ivcomp=p(1)
	else
	    ivcomp=1
	endif
	if(iflg2)cangle=p(2)
1401	continue
#ifndef unix
	IF(LOGITF)CALL LOGGER('PPLCONT')
#endif
	GOTO 6001
C     PLOT
1500	MAKEP='PLOT'
	GOTO 6002
C     READSEQ & RD
1600	CALL OPENF(FILE,IERR)
	IF(IERR.NE.0)GOTO 10
	CALL RDSEQ(X,NSIZE,PBUF,IBASE,FILE)
	GOTO 10
C     LIST
2900	CALL LIST(LTTOUT,X,NSIZE,LABEL)
	GOTO 10
C     SKPSEQ & SKP
3300	CALL OPENF(FILE,IERR)
	IF(IERR.NE.0)GOTO 10
	SYM='*PPL$EOF'
	CALL PUTSYM(SYM,NO,2,IER)
	J=IFIX(P(1))
	IF(M.EQ.0.OR.IFLG(1).EQ.0)J=1
	IF(IFORM(1:3).EQ.'DSF'.OR.IFORM(1:4).EQ.'BIBO'
     *	.OR.IFORM(1:3).EQ.'PPL')THEN
	DO 3304 I=1,J
	    CALL OPNDSF(FILE,'RD',LUNN)
	    CALL RDHDSF(LUNN,1,XPTS,K)
	    CALL CLSDSF(LUNN)
	    IF(K.EQ.0)GOTO 3302
3304	CONTINUE
	ELSE IF(IFORM(1:3).EQ.'EPI')THEN
	    CLOSE(LUNN)
	    DO 3305 I=1,J
		IF (LSCTD) THEN
		    CALL EPICRD(IDUM1,IDUM1,LEOF,.FALSE.)
		ELSE
		    CALL EPICTRD(IDUM1,XDUM1,XDUM2,XDUM3,XDUM4,XDUM5,
     *		    LEOF,.FALSE.)
		ENDIF
		IF(LEOF)GOTO 3302
3305	    CONTINUE
	ELSE IF(IFORM(1:3).NE.'UNF')THEN
	    DO 3301 I=1,J
3301	    READ(LUNN,*,END=3302)
	ELSE
	    DO 3303 I=1,J
3303	    READ(LUNN,END=3302)
	ENDIF
	GOTO 10
3302	IF(.NOT.QUIETF)WRITE(LTTOUT,'('' EOF'')')
	SYM='*PPL$EOF'
	CALL PUTSYM(SYM,YES,3,IER)
	GOTO 10
C	VIEW
3400	MAKEP='VIEW'
	IF(IFLG2)THEN
	    XYSCLE=1
	ELSE
	    XYSCLE=P(2)
	ENDIF
	IF(IFLG1)THEN
	    ZSCLE=P(1)
	ELSE
	    ZSCLE=0.
	ENDIF
	IF(M.GE.4.AND.(IFLG(3).NE.0.AND.IFLG(4).NE.0))THEN
	    VEWZMN=P(3)
	    VEWZMX=P(4)
	    IF(ZMN.GT.ZMX)THEN
		P1=VEWZMN
		VEWZMN=VEWZMX
		VEWZMX=P1
	    ENDIF
	ELSE
	    VEWZMN=ZMIN
	    VEWZMX=ZMAX
	ENDIF
	if(m.ge.5.and.iflg(5).ne.0)then
	    ivcomp=p(5)
	else
	    ivcomp=1
	endif
	GOTO 1401
C	ENTER
3500	IF(.NOT.QUIETF)WRITE(LTTOUT,'('' ENTER X,Y OR "END"'')')
	JTYPE=1
	YMIN=1.E38
	XMIN=YMIN
	XMAX=-XMIN
	YMAX=XMAX
	LINEN=LINEN+1
	CALL STPRMP('enter>',6)
	DO 3501 I=0,NSIZE-1
3503	    CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	    IF(ICOM.EQ.'END')GOTO 3502
	    X(IBASE+I)=P(1)
	    X(NSIZE/2+IBASE+I)=P(2)
	    IF(P(1).GT.XMAX)XMAX=P(1)
	    IF(P(1).LT.XMIN)XMIN=P(1)
	    IF(P(2).GT.YMAX)YMAX=P(2)
	    IF(P(2).LT.YMIN)YMIN=P(2)
3501	CONTINUE
3502	LLENG(LINEN)=I
	NX=I
	NY=1
	CALL RDSEQ2(X,NSIZE,PBUF,IBASE,FILE)
	CALL STPRMP('ppl+>',5)
	GOTO 10
C     MULTPLT
4900	CALL MULT
	GOTO 10
C     PLOTV
5300	MAKEP='PLOTV'
5301	IF(IFLG(1).EQ.1)VANG=P(1)
	GOTO 6002
C	PLOTUV
5700	MAKEP='PLOTUV'
	GOTO 5301
6002	CONTINUE
#ifndef unix
	IF(LOGITF)CALL LOGGER('PPLPLOT')
#endif
	IF(PBUF.EQ.0)THEN
	    IF(.NOT.QUIETF)WRITE(LTTOUT,'('' BUFF EMPTY'')')
	    GOTO 10
	ELSE
	    LINEN=0
	    IBASE=1
	ENDIF
6001	IF(N.NE.0)LAB1=LABEL
	ICOUNT=ICOUNT+1
	IF(IMULT.NE.1)THEN
	    CALL PLTIT(X,NSIZE)
	ELSE
	    CALL MPLOT(X,NSIZE)
	    IF(IMCNT.LT.MNX*MNY)GOTO 10
#ifdef unix
            rewind(lmult)
#else
	    OPEN(LMULT,FILE=MULTFL,STATUS='OLD',
     *	    FORM='UNFORMATTED')
#endif
	    DO 6003 IMCNT=1,MNX*MNY
		READ(LMULT,END=6004)IPTS,MXBOT,MYLEF
		READ(LMULT,END=6004)XLABS,YLABS,RLABS,LTYPE,XLLABS,
     *		YLLABS,LLTYPE,RHLABS,USRLBS
		CALL GET(LMULT)
		IF(JTYPE.EQ.1)THEN
		    CALL GETXY(LMULT,X,NSIZE)
		ELSE
		    CALL GETZ(LMULT,X,NX,NY)
		ENDIF
		IF(IZTYP.NE.0)CALL GETZG(LMULT)
		CALL PLTIT(X,NSIZE)
6003	    CONTINUE
#ifdef unix
 6004         CLOSE(LMULT)
#else
6004	    CLOSE(LMULT,DISP='DELETE')
#endif
c
c	show mult file is closed by clearing the file name
c
	    MULTFL=' '
	    CALL MPLOT2(X,NSIZE)
	ENDIF
	GOTO 10
C	VELVCT
9100	MAKEP='VELVCT'
	GOTO 5301
C	VECTOR
9110	MAKEP='VECTOR'
	IF(IFLG1)VSKPX=P(1)
	IF(IFLG2)VSKPY=P(2)
	GOTO 6001
C	VECSET
9120	IF(IFLG1)VLEN=P(1)
	IF(IFLG2)VUSRLN=P(2)
	VAUTO=M.LT.2
	GOTO 10
C	VECKEY
9130	IF(IFLG1)VXLOC=P(1)
	IF(IFLG2)VYLOC=P(2)
	IF(M.GE.3.AND.IFLG(3).NE.0)VPOS=P(3)
	IF(N.GT.0)VFRMT=LABEL
	VUSER=SUSER
	VKEY=M.GE.2
	GOTO 10
c	usr
9140	temp2=' '
	call pplusr(label,ier,temp2)
	IF(ier.ne.0)then
	    ilen=lnblk(temp2,80)
	    temp='USR: '//temp2(:ilen)
	    if(.not.quietf)WRITE(LTTOUT,'(a)')temp(:ilen+5)
	    temp2='C '//temp(:ilen+5)
	    call echo(temp2,ilen+7)
	endif
	goto 10
C	SHADE
9150    MAKEP = 'SHADE'                                                       
        IF(N.NE.0)LAB1=LABEL                                                  
        ICOUNT=ICOUNT+1                                                       
        IF(IMULT.NE.1)THEN                                                    
            CALL PLTIT(X,NSIZE)                                               
        ELSE                                                                  
            CALL MPLOT(X,NSIZE)                                               
            IF(IMCNT.LT.MNX*MNY)GOTO 10                                       
            OPEN(LMULT,FILE=MULTFL,STATUS='OLD',                              
     *      FORM='UNFORMATTED')                                               
            DO 9151 IMCNT=1,MNX*MNY                                           
            READ(LMULT,END=9152)IPTS,MXBOT,MYLEF                              
	    READ(LMULT,END=9152)XLABS,YLABS,RLABS,LTYPE,XLLABS,YLLABS,LLTYPE, 
     *	    RHLABS,USRLBS                                                     
            CALL GET(LMULT)                                                   
            IF(JTYPE.EQ.1)THEN                                                
                CALL GETXY(LMULT,X,NSIZE)                                     
            ELSE                                                              
                CALL GETZ(LMULT,X,NX,NY)                                      
            ENDIF                                                             
            IF(IZTYP.NE.0)CALL GETZG(LMULT)                                   
            CALL PLTIT(X,NSIZE)                                               
9151        CONTINUE    
#ifdef FORTRAN_90                                           
9152        CLOSE(LMULT,STATUS='DELETE')                             
#else
9152        CLOSE(LMULT,DISP='DELETE')                             
#endif
c                                                                  
c       show mult file is closed by clearing the file name         
c                                                                  
            MULTFL=' '                                             
            CALL MPLOT2(X,NSIZE)                                   
        ENDIF                                                      
        GOTO 10                                                    
C       SHAKEY                                                     
9160    call shade_key                                             
        goto 10                                                    
C       SHAKEY                                                     
9170    call shade_set                                             
        goto 10                                                    
c       area
*!9171 makep='AREA' NAME CHANGE *jd*
9171   makep='FILL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
 9172	makep='PLOTPOL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
 9173	makep='FILLPOL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
#ifdef unix
9180    continue
#else
9180    call pixmap                                             
#endif
        goto 10                                                    
9190	call aline (x,nsize)
	goto 10
9191    call set_one_color
        goto 10
c sqfill
 9380   if(labon)sqfflg=.true.
        if(laboff .or. n.eq.0)sqfflg=.false.
        goto 10
9999	RETURN

	END
	SUBROUTINE OPENF(FILE,IER)
c  If format is EPIC:
c  Opens EPIC file on unit=11 if label is not blank.
c  Returns logical variable lepic in COMMON/COMEPL/ as true or false
c  to indicate whether this is an EPIC file or simply a pointer file.
c
c  Programmed by N Soreide, Jun 85.
c  Modified for pointer file, Oct 85.
c  Patched up Feb 88
c
	character line*132

*       Declarations added 9.6.88 J Davison          
        integer ier,isym,lnblk,i,ix                  
                                                     
        external lnblk                               
*       End declarations 9.6.88                      

c
c
c  Get EPIC (RIM) file with data file names
c
#ifdef unix
	include 'tmap_pplv11inc/comepl_inc.decl'
	include 'pplv11inc/COMEPL.INC'
	include 'tmap_pplv11inc/epiclun_inc.decl'
	include 'pplv11inc/EPICLUN.INC'
	include 'tmap_pplv11inc/data_inc.decl'
	include 'pplv11inc/DATA.INC'
	include 'tmap_pplv11inc/cmrd_inc.decl'
	include 'pplv11inc/CMRD.INC'
	include 'tmap_pplv11inc/cmrdl_inc.decl'
	include 'pplv11inc/CMRDL.INC'
	include 'tmap_pplv11inc/lunits_inc.decl'
	include 'pplv11inc/LUNITS.INC'
	include 'tmap_pplv11inc/system_inc.decl'
	include 'pplv11inc/SYSTEM.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:COMEPL_INC.DECL'
	INCLUDE 'PPLV11INC:COMEPL.INC'
	INCLUDE 'TMAP_PPLV11INC:EPICLUN_INC.DECL'
	INCLUDE 'PPLV11INC:EPICLUN.INC'
	INCLUDE 'TMAP_PPLV11INC:DATA_INC.DECL'
	INCLUDE 'PPLV11INC:DATA.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRD_INC.DECL'
	INCLUDE 'PPLV11INC:CMRD.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRDL_INC.DECL'
	INCLUDE 'PPLV11INC:CMRDL.INC'
	INCLUDE 'TMAP_PPLV11INC:LUNITS_INC.DECL'
	INCLUDE 'PPLV11INC:LUNITS.INC'
	INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
	INCLUDE 'PPLV11INC:SYSTEM.INC'
#endif
	CHARACTER FILE*(*),SYM*30
	IER=0
	IF(N.EQ.0)THEN
	    IF(FILE.EQ.' ')THEN
		IF(.NOT.QUIETF)WRITE(LTTOUT,999)
999		FORMAT(' NO FILE')
		IER=1
	    ELSE IF(IFORM(1:3).EQ.'EPI')THEN
		CLOSE(LUN1)
	    ENDIF
	ELSE
	    SYM='*PPL$INPUT_FILE'
	    ISYM=LNBLK(LABEL,255)
	    CALL PUTSYM(SYM,LABEL,ISYM,IER)
	    CLOSE(LUNN)
#ifdef unix
            FILE=LABEL
#else
	    I=INDEX(LABEL,':')
	    IF(I.LE.0)THEN
		FILE='SY:'
		FILE(4:)=LABEL
	    ELSE
		FILE=LABEL
	    ENDIF
#endif
	    IF(IFORM(1:3).EQ.'DSF'.OR.IFORM(1:3).EQ.'BIB'
     *	    .OR.IFORM(1:3).EQ.'PPL')THEN
	        RETURN
	    ELSE IF(IFORM(1:3).EQ.'EPI')THEN
C
c  		Open EPIC file to read data file names
c
	    CLOSE(LUN1)
	    CLOSE(LUN11)
#ifdef unix
            open (unit=LUN11, form='formatted', status='old',
     1      file=FILE,err=100 )
#else
open (unit=LUN11, form='formatted', status='old',
     1	    readonly,file=FILE,err=100 )
#endif
c
c  Check whether this is EPIC file or other pointer file.
c
	    read (LUN11, 102) line
102	    format (a)
	    ix = index (line(1:132), 'EPIC')
	    if (ix .ne. 0 .or. line(1:1).eq. ' ')then
		lepic = .true.
	    else
		lepic = .false.
	    endif
	    rewind (LUN11)
	ELSE IF(IFORM(1:3).NE.'UNF')THEN
#ifdef unix
            OPEN(UNIT=LUNN,FILE=FILE,STATUS='OLD',ERR=100)
        ELSE
            OPEN(UNIT=LUNN,FILE=FILE,STATUS='OLD',ERR=100,
     *      FORM='UNFORMATTED')
#else
	    OPEN(UNIT=LUNN,FILE=FILE,STATUS='OLD',ERR=100,READONLY)
	ELSE
     	    OPEN(UNIT=LUNN,FILE=FILE,STATUS='OLD',ERR=100,READONLY,
     *	    FORM='UNFORMATTED')
#endif
	ENDIF
	ENDIF
	RETURN
100	IER=2
	IF(.NOT.QUIETF)WRITE(LTTOUT,998)FILE
998	FORMAT(1X,A30,' NOT FOUND')
	RETURN
	END
