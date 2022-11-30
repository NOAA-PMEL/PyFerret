	SUBROUTINE PPLCMD_F(ISI,ICMDIM,ICMSZE,pplmem)

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
*	J Davison 10.15.98 Mod for pattern support: PATSET
*       *jd* VVV  02.16.99 Mod to bring two versions of PPLCMD in line VVV
*	S Hankin  11.25.97 Mod to add commands PLOTPOL and FILLPOL
*       kob 9.98 - add external statement to force linker to find
*                  block data statments
*       J Davison 10.14.99 Mod to add CALENDAR command for multiple calendars
* V510: *sh* 3/0 - bug fix: TXLABP 0 was interpreted as TXLABP 1 cuz
*		of SIGN() call near top of this routine
*       ACM 10/2000  Add VECTOR/FLOWLINE, alias FLOWLINE 
* V530: *acm* 1/01 calendar info in calendar.cmn and calendar.decl
* V54:  *acm* 9/01 changes for dynamic allocation of PPLUS memory.
*                  This routine called by a C "envelope" routine PPLCMD
*                  with the same arguments as the original PPLCMD.F, which
*                  gets the global pointer to PPLUS memory and passes it to
*                  this routine.   The only change here is that array pplmem
*                  (old name was X)  is passed in rather than in PPLDAT.INC
*      *acm* 10/01  remove VMS includes
* v541 *acm*  5/02 add option to control outline on the shade key boxes (S_KEY)
* v552 *acm*  3/03 Add poly
* v552 *acm*  3/03 Longer strings: changes for multi-line labels
* v552 *acm*  5/03 Add option to control whether cellarray calls are made for
*                  shade plots (S_CELL)  Only used internally.
* v580 *acm* 8/04 Change use of 256 to buflen+1, 
*                 where buflen is defined in cmndlen.inc
* v580 *acm* 8/04 allow PPL POLYGON as well as PPL FILLPOL after POLY/SET
* V581 *acm* 4/05 fix bug 1214; clsplt needs to set meta_actv to false.
* V581 *acm* 5/05 Add simple error checking to the text with XFOR, YFOR, and VECKEY
*                 to see if they are a valid fortran format strings(bug1215). Note
*                 this is just a first-pass checking: if mode ignore_error is on,
*                 or if a plot command is issued without correcting the PPL call,
*                 the bad format string is passed on to routines like numcde or
*                 vecfld, where it will cause a crash.
C *acm* 11/05     new options for XFOR, YFOR, 'DM' deg/min deg/min
* V600: *acm* 11/05 add xdms,ydms for deg/min labels
C        1/06  xvaloff,yvaloff add this amount to each x or y axis label. 
C              (axis data was read as double precision and converted to single 
C              after applying an offset.)
C V600: *acm* 4/06 Change how RANGE_DM is applied when using DM or DMS axis labelling.
C V600: *acm* 5/06 Allow seconds in DMS axis labelling; had removed this.
C V600 *acm* 5/2006 LINECOLORS User can change the number of line colors.
C V602  1/07 *acm* true batch mode for metafiles; flag meta_batch
C                 distinguises this case from other batch modes.
C V610 *acm* 11/2007 For build using PGI Fortran compilers at ECMWF 
C                 The code testing the Format for axis labels balks at
C                 writing a 0 using an F format. See changes at lines 
C                 WRITE(err_msg,label,ERR=1999) 0.  Need to check for an I
C                 format and write 0, else write 0.0.
C V63  *acm* 10/09 Changes for gfortran build
C V65  *acm*  1/10 New time-axis style for DAY and MONTH, DMY: adds the
C                  year to DAY,MONTH or to MONTH.
* V65   *acm* 2/10 Fix bug 1724: in metafile batch mode only, if some command
*                  has opened the metafile but nothing plotted yet, and if
*                  a SET MODE META:newname.plt is issued, just rename the metafile
*                  that was opened. New RENAMPL command.
C *acm*  3/10  Bug 670 fixed: New flag MULTICENFLAG for multi-decade plots, 
C              do not make small tics by default.
* V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* v691 5/14 *acm* Change name MOD to TEKMOD
* V6.92 *acm* 6/14 New color /key options: any of /key=horizontal,centerlabel
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V694  *acm*  5/15 Ticket 2283. Change to call for subroutine setax. 
* v696 *acm*	gap_var to define seperate line segments on PLOT/VS plots
*                  (LAS ticket 1597 and others)
* V695  *acm*  9/15 Changes for hour:min labeling on time axes
* v695 9/15 *acm* write values to PPL* symbols using 6 digits not 4
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, use its
*                   delta-t for time overlay plots. TAXUND restores the default.
* V698  4/16 *acm* New PLOT/NTH=
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* V741 *acm*  5/18 issue 1874: consistently apply /CONSET number of digits setting
*	*acm* 11/2022 Code cleanup. Issue messages and remove code for unused PPLUS commands:
*		   READSEQ, RD, SAVE, GET, ENTER, MUTLPLT, TAXUND, PLTYPE (can only be GKS)
*		   TEKNME, ECHO
* 11/2022 *acm* Code cleanup: remove references to obsolete gui interface
* 11/2022 *acm* Code cleanup: Use the variable from common CMXGE rather than 1.e35
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.

c	fromi	command file or device (/dev/tty or TT: for terminal)
c	linei	input line for parameters
c	isi	length of line in characters
c	combuf() command buffer for executing commands from memory
c	icmdim	dimensioned length of combuf
c	icmsze	number of lines in combuf
c
        include 'comepl_inc.decl'  ! added 9.6.88 j davison 
        include 'COMEPL.INC'  ! added 9.6.88 j davison 
	include 'comeps_inc.decl'
	include 'COMEPS.INC'
	include 'comepv_inc.decl'
	include 'COMEPV.INC'
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'PPLDAT.INC'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'pen_inc.decl'
	include 'PEN.INC'
	include 'labels_inc.decl'
	include 'LABELS.INC'
	include 'labcom_inc.decl'
	include 'LABCOM.INC'
	include 'cont_inc.decl'
	include 'CONT.INC'
	include 'dashzz_inc.decl'
	include 'DASHZZ.INC'
	include 'conlab_inc.decl'
	include 'CONLAB.INC'
	include 'plt_inc.decl'
	include 'PLT.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
	include 'hab_inc.decl'
	include 'HAB.INC'
	include 'data_inc.decl'
	include 'DATA.INC'
	include 'cmrd_inc.decl'
	include 'CMRD.INC'
	include 'cmrdl_inc.decl'
	include 'CMRDL.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'mplot_inc.decl'
	include 'MPLOT.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'tics_inc.decl'
	include 'TICS.INC'
	include 'tics2_inc.decl'
	include 'TICS2.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
	include 'xylabp_inc.decl'
	include 'XYLABP.INC'
	include 'view_inc.decl'
	include 'VIEW.INC'
	include 'zgrid_inc.decl'
	include 'ZGRID.INC'
	include 'filnam_inc.decl'
	include 'FILNAM.INC'
	include 'lunits_inc.decl'
	include 'LUNITS.INC'
	include 'vector_inc.decl'
	include 'VECTOR.INC'
	include 'system_inc.decl'
	include 'SYSTEM.INC'
	include 'cmdcom_inc.decl'
	include 'CMDCOM.INC'
	include 'symkey_inc.decl'
	include 'SYMKEY.INC'
	include 'switch_inc.decl'
	include 'SWITCH.INC'
	include 'prmcom_inc.decl'
	include 'PRMCOM.INC'
	include 'gkscm2.cmn'
        include 'ARFILL.INC'
	include 'calendar.decl'
	include 'calendar.cmn'
        INCLUDE 'pplcmd_strings.cmn'
        INCLUDE 'cmndlen.inc'
        INCLUDE 'errmsg.parm'

* special COMMON to let this routine know if FERRET is in control
        include 'ppl_in_ferret.cmn'
                                     
        real            pplmem(*)
        logical         tmapdebug
CCCCCCC Assign value below rather than initialize here.  F90 complains (linux)
CCCCCCC data	        tmapdebug/.false./

        common/tmdebug/ tmapdebug

*       Declarations added J Davison 9.2.88                    
        integer         icmdim,is,isi,icmsze,i,ier,isym        
        integer         j,k,ic,ierr,lun1,lun11                 
        integer         idum1,iy1,ix1,ix3                   
        integer         lnblk,ws_state

	REAL*8		DIFF
        real            xm,ym,xt,xdum1,xdum2,xdum3,xdum4       
        real            xdum5,zmn,zmx,p1,x1,y1,y2,x3,x4     
        real            symwid,xmx,xmn,ymx,ymn,dd 
                                                               
        external        lnblk,diff,symwid                      
                                                               
        logical         first_time_thru               
        data            first_time_thru/.true./                

*       End declarations 9.2.88
                                
*	Declarations added J Davison 11.30.88
	real		xft,yft,xlt,ylt,xht,yht,xtit,xtlt
	integer		nmxt,nmyt,ilen
*	End declarations 11.30.88

	integer	  ipen_save(0:nlines), cal_status
	data	  ipen_save/1,nlines*1/
        INTEGER   TM_LENSTR, TM_LENSTR1, slen

	character strr*2048
        LOGICAL TM_HAS_STRING
	CHARACTER DTE*10,TEMP*2048,TEMP2*2048,FILE*81,FROM*80,LINE*2048
	CHARACTER SYM*120,ICOM7*7
	CHARACTER FROMI*10,LINEI*10
	CHARACTER COMBUF(ICMDIM)*2048
	CHARACTER YES*3,NO*2,err_msg*40
	LOGICAL LEOF
	LOGICAL IFLG1,IFLG2,LABON,VALOFF,TBOTH
	DIMENSION X1(5),Y1(5),Y2(4),X3(2),IX3(5)
	DIMENSION X4(4)
	INTEGER	  num_comm
	PARAMETER(num_comm = 126)
	CHARACTER COMM(num_comm)*7
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
     *  'TXLINT','TXNMTC','TXLABP','TAXUND','AUTOLAB','EVAR','DEBUG','RESET',
     *  'TITLE','VPOINT','CLSPLT','VECTOR','VECSET','VECKEY','USR',
     *  'SHADE','SHAKEY','SHASET','PIXMAP','ALINE','TMDEBUG','COLOR',
     *  'SQFILL','FILL', 'PLOTPOL', 'FILLPOL', 'PATSET','CALENDA',
     *  'FLOWLIN', 'S_KEY', 'S_CELL', 'POLYGON', 'XVALOFF', 'YVALOFF',
     *  'LINECOL','RENAMPL','SHRINKY','RIBBON','RIBFAST','RIBMISS',
     *  'GAPLOC','CLRKLAB'/

!  *  'SQFILL','AREA'/ NAME CHANGE *jd*
 
	CHARACTER*15  TMPCHR
	character*2048 set_name

*	*jd* linux external stmt below inserted	
	external scale

* *kob* add external statment 9/98
	EXTERNAL 
     .          PPLDAT,
     .          WS_TYPES_INIT

        include 'tmap_errors.parm'
        include 'century.cmn'

*****************************************************************************

*  get strings out of common.
      DO i = 1, ICMDIM
        COMBUF(i) = PPLCOMBUF(i) 
      ENDDO

      FROMI = PPLFROMI
      LINEI = PPLLINEI

*       SETUP SHADE ROUTINE DEFAULTS             
        if (first_time_thru) then                
	   tmapdebug = .false.
           first_time_thru = .false.             
           call setup_defaults                   
        end if                                   

C
C	SET UP LABELT STUFF
C
	IF ( LABELT .EQ. 0 ) THEN
	   ITXLBP= 0		! 3/00 bug fix for "TXLABP 0"
	ELSE
	   ITXTPE=ABS(LABELT)
	   ITXLBP=SIGN(1,LABELT)
	ENDIF
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
	      CMDBUF(I)=COMBUF(I)
30	    CONTINUE
	    FROM='$$MEMBUF$$'
	    LINE=' '
	    IS=0
	ENDIF
        TERMF=FROM.EQ.'/dev/tty'
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
10	call FGD_GQOPS(ws_state)
!	If GKS WS is active, flush output buffer before leaving if just plotted
	if (ws_state .ge. 3) then
	  if (makep .eq. 'CONTOUR'
     .	  .or.makep .eq. 'PLOT'
     .	  .or.makep .eq. 'VIEW'
     .	  .or.makep .eq. 'PLOTV'
     .	  .or.makep .eq. 'PLOTUV'
     .	  .or.makep .eq. 'VELVCT'
     .	  .or.makep .eq. 'VECTOR'
     .	  .or.makep .eq. 'FLOW'
     .	  .or.makep .eq. 'SHADE'
     .	  .or.makep .eq. 'FILL'
     .	  .or.makep .eq. 'PLOTPOL'
     .	  .or.makep .eq. 'FILLPOL'
     .	  .or.makep .eq. 'POLYGON'
     .	  .or. icom(1:1) .eq. '%') then
             IF ( PPL_in_FERRET ) CALL SEG_ON( )
             call pplgflush
             IF ( PPL_in_FERRET ) CALL SEG_OFF
	   endif
	endif
	makep = ' '
        IF (makep .EQ. 'POLYGON') makep = 'FILLPOL'

	CALL GETCMD(STRING,ISLEN,ICOM,P,IFLG,M,LABEL,N,LBSTRT)
	IF (DONEF) RETURN

C       Changed ICOM to CHAR*7 comparison for CALENDAR command. *jd* 10.99
	ICOM7(1:7)=ICOM(1:7)
	DO 20 I=1,num_comm
	    IF(ICOM7.EQ.COMM(I))GOTO 21
20	CONTINUE
	IF(ICOM(1:1).EQ.'%')THEN
! 	   Mod 11.9.93 to insert advanced command output into segments
*          TMAP check that output window is active
           IF ( PPL_in_FERRET ) CALL CK_GKS_ACTIVE( *9999 ) 
*          TMAP addition to turn on segment
           IF ( PPL_in_FERRET ) CALL SEG_ON( )

	   CALL PRMTIV(pplmem,pplmem_nsize)
           IF ( PPL_in_FERRET ) CALL SEG_OFF
	ELSE
	    CALL SPWN
	ENDIF
	GOTO 10
21	IFLG1=(M.GE.1.AND.IFLG(1).NE.0)
	IFLG2=(M.GE.2.AND.IFLG(2).NE.0)
	LABON=(LABEL.EQ.'ON'.OR.LABEL.EQ.'on')
	TBOTH=(LABEL.EQ.'BO'.OR.LABEL.EQ.'bo')
	IF (TBOTH) LABON = .TRUE.
	VALOFF=(INDEX(LABEL,'OFF')+INDEX(LABEL,'off')).NE.0
C       Mod below for brain dead version *JD* 8.2.91
        if (tmapdebug) then
	GOTO (100,200,300,500,600,700,900,1000,1100,1200,1300,
     *	1400,1500,1600,1800,1900,2000,2100,2200,2300,2500,2600,
     *	2700,2800,2900,3300,3400,3500,3600,4000,4200,4300,
     *	4400,4500,4600,4700, 600,1600,3300,4900,5000,5100,5200,
     *	5300,5400,5500,5700,5800,5900,6000,6100,6200,6300,6400,
     *	6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,
     *	7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,
     *	8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9750,
     *	9800,9900,9910,9920,9930,9940,9950,9110,9120,9130,9140,
     *	9150,9160,9170,9180,9190,98,9191,9380,9171,9172,9173,
     *  9174,9175,9390,9165,9166,9183,4401,4501,9192,9960,9970,
     *  1550,9980,9982,9984,9985),I
        else
	GOTO (100,200,300,500,99,700,900,1000,1100,1200,1300,
     *	1400,1500,99,1800,1900,2000,2100,2200,2300,2500,2600,
     *	2700,2800,2900,99,3400,3500,3600,4000,4200,4300,
     *	4400,4500,4600,4700,99,99,99,4900,5000,5100,5200,
     *	5300,5400,5500,5700,5800,5900,6000,6100,6200,6300,6400,
     *	6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,
     *	7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,
     *	8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9750,
     *	9800,9900,9910,9920,9930,9940,9950,9110,9120,9130,9140,
     *	9150,9160,9170,9180,9190,98,9191,9380,9171,9172,9173,
     *  9174,9175,9390,9165,9166,9183,4401,4501,9192,9960,9970,
     *  1550,9980,9982,9984,9985),I
        end if
        goto 10

C     TMDEBUG
 98   call upper (label,2048)
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
200	CONTINUE  ! unused
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
	GOTO 10
600	CONTINUE ! unused
C     SIZE
700	WIDTH=P(1)
	HEIGHT=P(2)
	SYM='*PPL$WIDTH'
	CALL PUTVAL(SYM,WIDTH,6,IER)
	SYM='*PPL$HEIGHT'
	CALL PUTVAL(SYM,HEIGHT,6,IER)
	GOTO 10
C     SMOOTH
900	IF(M.EQ.0)GOTO 10
	CALL SMOOTH(pplmem,NX,NY,NX,NY,IFIX(P(1)))
	GOTO 10
C     TKTYPE
1000	CONTINUE

	print *, 'PPLUS TKTYPE command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

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

C     LABS
2300	IF(P(1).EQ.0..OR.M.EQ.0)THEN
	    DO 2301 I=1,NLABS
		LLTYPE(I)=0
		RHLABS(I)=-1.0
2301	    CONTINUE
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
		XLABS(I)=cmxge
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
	  LWGT(I)=1
2702	CONTINUE
	GOTO 10
C     CONSET
2800	IF(IFLG1)HGT=P(1)
	NEWSIG = 0  ! set this flag if a PPL CONSET is specificially setting the nsig value
	IF(IFLG2)THEN
	   IF (IFIX(P(2)) .GT. 0) NEWSIG = 1
	   NSIGC=IFIX(P(2))
	ENDIF
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

C     SAVE
4200	CONTINUE

	print *, 'PPLUS SAVE command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

	GOTO 10
C	GET
4300	CONTINUE

	 print *, 'GET  '//icom(:TM_LENSTR1(icom))
         err_msg = 'PSAVEPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

	GOTO 10


c For XFOR and YFOR, check if LABEL is a valid format: 
c do a test write to the string err_msg, using label as a format
c If format is (DM) then axis will be labelled with 
c deg/min, or deg/min.ff decimal fraction of minutes.
c****If it is XFOR(DM,ddd) then we will rotate the labes by ddd degrees.****

C   RANGE_DM recomputes axis range and delta, so that it has
C   good intervals when expressed as deg-min-sec (A previous 
C   call to RANGE sets intervals based on degrees and decimal 
C   fractions of degrees.) Reset only the delta, as the user may
C   have explicitly set the lo and hi limit

C     XFOR
4400  i = STR_UPCASE (TMPCHR, LABEL(1:5))
      i = TM_LENSTR1(LABEL) - 1
      IF (TMPCHR(2:2) .EQ. 'D') THEN  ! degrees/mintues/seconds ?
           IF (TMPCHR(2:3) .EQ. 'DM') THEN
              xdms = 0
              IF (TMPCHR(2:3) .EQ. 'DM') xdms = 1 
              IF (TMPCHR(2:4) .EQ. 'DMS') xdms = 2 
              IF (xdms .GT. 0) THEN
	         CALL RANGE_DM(XLO,XHI,XTIC,xmn,xmx,dd)
                 xtic = dd
	      ENDIF
           ELSE ! 'DD' resets to decimal degrees
              xdms = 0
           ENDIF
           GOTO 10
        ENDIF

        IF (TMPCHR(2:4) .EQ. 'SPC') THEN  ! Space between degree-sign and EW or NS
           xdspac = 0
           IF (TMPCHR(5:5) .NE. ')') READ (TMPCHR(5:5), *) xdspac
           GOTO 10
        ENDIF

        IFRX=LABEL
        slen = TM_LENSTR(LABEL)
        IF (slen .GT. 1) THEN
           IF (TM_HAS_STRING( label, '(I')) THEN
              WRITE(err_msg,label,ERR=1999) 0
           ELSE
              WRITE(err_msg,label,ERR=1999) 0.0
           ENDIF
        ELSE
	   xdspac = 0
	   xdms = 0
        ENDIF
	GOTO 10
C     YFOR

4500  i = STR_UPCASE (TMPCHR, LABEL(1:15))
      i = TM_LENSTR1(LABEL) - 1
      IF (TMPCHR(2:2) .EQ. 'D') THEN  ! degrees/mintues/seconds ?
           IF (TMPCHR(2:3) .EQ. 'DM') THEN
              ydms = 0
              IF (TMPCHR(2:3) .EQ. 'DM') ydms = 1 
              IF (TMPCHR(2:4) .EQ. 'DMS') ydms = 2 
              IF (ydms .GT. 0) THEN
	         CALL RANGE_DM(YLO,YHI,YTIC,ymn,ymx,dd)
                 ytic = dd
	      ENDIF

           ELSE ! 'DD' resets to decimal degrees
              ydms = 0
           ENDIF
           GOTO 10
        ENDIF

        IF (TMPCHR(2:4) .EQ. 'SPC') THEN  ! Space between degree-sign and EW or NS
           ydspac = 0
           IF (TMPCHR(5:5) .NE. ')') READ (TMPCHR(5:5), *) ydspac
           GOTO 10
        ENDIF

        IFRY=LABEL
        slen = TM_LENSTR(LABEL)
        IF (slen .GT. 1) THEN
           IF (TM_HAS_STRING( label, '(I')) THEN
              WRITE(err_msg,label,ERR=1999) 0
           ELSE
              WRITE(err_msg,label,ERR=1999) 0.0
           ENDIF
        ELSE
	   ydspac = 0
	   ydms = 0
        ENDIF
	GOTO 10
        
C     XVALOFF
4401  XVALOFF=P(1)
      i = XVALOFF
      IF (FLOAT(i) .NE. XVALOFF) THEN
         err_msg = 'Offset must be an integer'
         CALL ERRMSG (ferr_out_of_range, status, err_msg, *9999)
      ENDIF
      GOTO 10

C     YVALOFF
4501  YVALOFF=P(1)
      i = YVALOFF
      IF (FLOAT(i) .NE. YVALOFF) THEN
         err_msg = 'Offset must be an integer'
         CALL ERRMSG (ferr_out_of_range, status, err_msg, *9999)
      ENDIF
      GOTO 10

C     ORIGIN
4600	IF(IFLG1)XORG=P(1)
	IF(IFLG2)YORG=P(2)
	SYM='*PPL$XORG'
	CALL PUTVAL(SYM,XORG,6,IER)
	SYM='*PPL$YORG'
	CALL PUTVAL(SYM,YORG,6,IER)
	if(pltopn)then
	    call setax(pplmem,pplmem_nsize,xft,yft,xlt,ylt,xht,yht,
     .                 nmxt,nmyt,xtit,xtlt,.FALSE.)
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
5200	CALL UPPER(LABEL,2048)
	IF(INDEX(LABEL,'XLE').NE.0)THEN
	    XLE=.NOT.VALOFF
	    IF(IFLG1)CMXLE=P(1)
	ELSE IF(INDEX(LABEL,'XEQ').NE.0)THEN
	    XEQ=.NOT.VALOFF
	    IF(IFLG1)CMXEQ=P(1)
	ELSE IF(INDEX(LABEL,'XGE').NE.0)THEN
	    XGE=.NOT.VALOFF
	    IF(IFLG1)CMXGE=P(1)
	ELSE IF(INDEX(LABEL,'YLE').NE.0)THEN
	    YLE=.NOT.VALOFF
	    IF(IFLG1)CMYLE=P(1)
	ELSE IF(INDEX(LABEL,'YEQ').NE.0)THEN
	    YEQ=.NOT.VALOFF
	    IF(IFLG1)CMYEQ=P(1)
	ELSE IF(INDEX(LABEL,'YGE').NE.0)THEN
	    YGE=.NOT.VALOFF
	    IF(IFLG1)CMYGE=P(1)
	ELSE IF(INDEX(LABEL,'ZLE').NE.0)THEN
	    ZLE=.NOT.VALOFF
	    IF(IFLG1)CMZLE=P(1)
	ELSE IF(INDEX(LABEL,'ZEQ').NE.0)THEN
	    ZEQ=.NOT.VALOFF
	    IF(IFLG1)CMZEQ=P(1)
	ELSE IF(INDEX(LABEL,'ZGE').NE.0)THEN
	    ZGE=.NOT.VALOFF
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
	IF (TBOTH) ITFLG=2
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
* The following lets the user plot small tics with PPL TICS
	MULTICENFLAG = .FALSE.
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
6201	GOTO 10

C	AXLEN
6300	IF(IFLG1)XLEN=P(1)
	IF(IFLG2)YLEN=P(2)
	SYM='*PPL$XLEN'
	CALL PUTVAL(SYM,XLEN,6,IER)
	SYM='*PPL$YLEN'
	CALL PUTVAL(SYM,YLEN,6,IER)
	if(pltopn)then
	    call setax(pplmem,pplmem_nsize,xft,yft,xlt,ylt,xht,yht,nmxt,
     *	    nmyt,xtit,xtlt,.FALSE.)
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
	    IF(IFLG(I).NE.0) IAXON(I)=P(I)
7501	CONTINUE
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
*	    ISYM=LNBLK(TEMP,2048)
*	    CALL PUTSYM(SYM,TEMP,ISYM,IER)
*           Open a "metafile"
*           *kms* First check if this is actually a rename
8000        IF ( save_on_exit .AND. imgname_sent ) THEN
               CALL FGD_SAVE_WINDOW(meta_file)
            ENDIF
            IF(N.EQ.0)THEN
                meta_file='ferret.png'
                TEMP='ferret.png'
	    ELSE
		meta_file=LABEL
		TEMP=LABEL
	    ENDIF
*           *kms* Mark that the graphics are to be saved
*                 and clear the active window
            save_on_exit = .TRUE.
            imgname_sent = .FALSE.
            CALL OPEN_METAFILE
	    SYM='*PPL$PLTNME'
	    ISYM=LNBLK(TEMP,2048)
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
	    CALL PUTVAL(SYM,XFCT(I),6,IER)
	    WRITE(SYM,997)'XOFF',I
	    CALL PUTVAL(SYM,XOFF(I),6,IER)
	    WRITE(SYM,997)'YFACT',I
	    CALL PUTVAL(SYM,YFCT(I),6,IER)
	    WRITE(SYM,997)'YOFF',I
	    CALL PUTVAL(SYM,YOFF(I),6,IER)
	    GOTO 10
C	MARKH
8500	    I=IFIX(P(1))
	    IF(IFLG2)HMARK(I)=P(2)
	    IF(M.EQ.1)HMARK(I)=0.08
	    IF(M.GE.3)nskpsym(I) = p(3)
	    GOTO 10
C	LINFIT
8600	    I=IFIX(P(1))
	    IF(I.GT.LINEN.OR.LLENG(I).LE.2.OR.LINEN.GE.NLINES)THEN
		IF(.NOT.QUIETF)WRITE(LTTOUT,
     *            '('' LINFIT ERROR -- NO ROOM OR LINE'',I3,
     *            '' DOES NOT EXIST'')')
		GOTO 10
	    ENDIF
	    CALL LINFIT(I,pplmem,pplmem_nsize,IBASE,PBUF)
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
8800	CONTINUE 

	print *, 'PPLUS GRID command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

C	TEKNME
8900	CONTINUE

	print *, 'PPLUS TEKNME command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

C	ECHO
9000	CONTINUE

	print *, 'PPLUS ECHO command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

C	TIME
9200	CONTINUE

	IF(N.EQ.0)THEN
		IAUTOT=1
		GOTO 10
	    ENDIF
	    IAUTOT=0

	    IF (itflg .EQ. 2) GOTO 9230

C	GET TMIN,TMAX,TSTART FROM LABEL
C
C	IF NO FIRST PARAMETER SKIP TMIN PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9201
	    IC=INDEX(LABEL,',')
	    IF(IC.EQ.0)IC=buflen+1
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
		IF(IS.GT.2048)GOTO 9203
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
	    IF(IC.EQ.0)IC=buflen+1
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
		IF(IS.GT.2048)GOTO 9203
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
c------------------------------------------------------------------------------------

C Time definition for second time axis (forecast/time plot)
9230    CONTINUE

C	GET TMIN,TMAX,TSTART FROM LABEL
C
C	IF NO FIRST PARAMETER SKIP TMIN PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9231
	    IC=INDEX(LABEL,',')
	    IF(IC.EQ.0)IC=buflen+1
	    IS=INDEX(LABEL,' ')
	    IS=MIN0(IS,IC)

	    TMPCHR = LABEL(1:IS-1)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		IT1b=TMPCHR(2:11)         ! old format
	 	IT1b(11:14) = '0019'
	    ELSE
		IT1b(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (tmpchr(11:12) .EQ. ' ') IT1b(11:12)='00' 
		IT1b(13:14) = TMPCHR(2:3) ! century
	    ENDIF

	    IF(ITSTRTb.EQ.' ')ITSTRTb=IT1b
	    IF(IC.NE.0)IS=IS+1
9240	    IF(LABEL(IS:IS).EQ.' ')THEN
		IS=IS+1
		IF(IS.GT.2048)GOTO 9233
		GOTO 9240
	    ELSE IF(LABEL(IS:IS).EQ.',')THEN
		LABEL=LABEL(IS+1:)
		GOTO 9232
	    ELSE
		LABEL=LABEL(IS:)
	    ENDIF
C
C	IF NO SECOND PARAMETER SKIP TMAX PROCESSING
C
9231	    IF(M.GE.2.AND.IFLG(2).EQ.0)GOTO 9232
	    IC=INDEX(LABEL,',')
	    IF(IC.EQ.0)IC=buflen+1
	    IS=INDEX(LABEL,' ')
	    IS=MIN0(IS,IC)

	    TMPCHR = LABEL(1:IS-1)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		IT2b=TMPCHR(2:11)         ! old format
		IT2b(11:14) = '0019'
	    ELSE
		IT2b(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (TMPCHR(11:12) .EQ. ' ') IT2b(11:12)='00' 
		IT2b(13:14) = TMPCHR(2:3) ! century
	    ENDIF

	    IF(IC.NE.0)IS=IS+1
9250	    IF(LABEL(IS:IS).EQ.' ')THEN
		IS=IS+1
		IF(IS.GT.2048)GOTO 9233
		GOTO 9250
	    ELSE
		LABEL=LABEL(IS:)
	    ENDIF

9232	    TMPCHR = LABEL(1:15)

	    IF ( TMPCHR(12:12) .EQ. ' ' ) THEN
		ITSTRTb=TMPCHR(2:11)         ! old format
		ITSTRTb(11:14) = '0019'
	    ELSE
		ITSTRTb(1:12)=TMPCHR(4:15)   ! enhanced format
		IF (TMPCHR(11:12) .EQ. ' ') ITSTRTb(11:12)='00' 
		ITSTRTb(13:14) = TMPCHR(2:3) ! century
	    ENDIF
	    


9233	    TLOb=1.-DIFF(IT1b,ITSTRTb)/DT
	    GOTO 10

c------------------------------------------------------------------------------------


C	TXTYPE
9300	CONTINUE	

C	IF NO FIRST PARAMETER SKIP PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9301
	    is_hours_axis = 0
	    IF(LABEL.EQ.'MIN')THEN
		ITYPET=0
		is_hours_axis = 2
	    ELSE IF(LABEL(1:1).EQ.'H')THEN
		ITYPET=0
		is_hours_axis = 1
	    ELSE IF(LABEL(1:1).EQ.'D')THEN
		ITYPET=0
		is_hours_axis = 0
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
		MONYRLAB = .FALSE.
		IF (INDEX(LABEL,'DMY').NE.0) MONYRLAB = .TRUE.
	    ELSE
		IF(INDEX(LABEL,'MON1').NE.0)THEN
		    ITXTPE=3
		ELSE IF(INDEX(LABEL,'MONY').NE.0)THEN
		    ITXTPE=2
		ELSE
		    ITXTPE=1
		ENDIF

*               Century axis mod
	        CENFLAG = .FALSE.
	        IF(INDEX(LABEL,'MULTIY').NE.0) CENFLAG = .TRUE.

*               multi-decade axis mod
	        MULTICENFLAG = .FALSE. 
	        IF(INDEX(LABEL,'MULTID').NE.0) THEN
		   CENFLAG = .TRUE. 
		   MULTICENFLAG = .TRUE. 
	        ENDIF
	    ENDIF
	    MONYRLAB = .FALSE.  ! Option for MON and DAY types. 
	    IF (INDEX(LABEL,'DMY').NE.0) MONYRLAB = .TRUE.

c Do not automatically turn on Day-Month-Year lables for shorter time axes
c            IF (LABEL.EQ.'HR' .OR. LABEL.EQ.'DAY') MONYRLAB = .TRUE.
	    LABELT=ITXTPE*ITXLBP
	    GOTO 10

	
c------------------------------------------------------------------------------------
9330	    CONTINUE
C
C	IF NO FIRST PARAMETER SKIP PROCESSING
C
	    IF(M.GE.1.AND.IFLG(1).EQ.0)GOTO 9331
	    is_hours_axis = 0
	    IF(LABEL.EQ.'PDAY')THEN
		ITYPETb=0
		is_hours_axis = 2
	    ELSE IF(LABEL(1:1).EQ.'H')THEN
		ITYPETb=0
		is_hours_axis = 1
	    ELSE IF(LABEL(1:1).EQ.'D')THEN
		ITYPETb=0
		is_hours_axis = 0
	    ELSE IF(LABEL(1:1).EQ.'M')THEN
		ITYPETb=1
	    ELSE
		ITYPETb=3
	    ENDIF
C
C	PROCESS TIME AXIS STYLE
C
9331	    IF(ITYPETb.EQ.0)THEN
		IF(INDEX(LABEL,'HRD').NE.0)THEN
		    ITXTPEb=2
		ELSE
		    ITXTPEb=1
		ENDIF
	    ELSE IF(ITYPETb.EQ.1)THEN
		IF(INDEX(LABEL,'DAYM').NE.0)THEN
		    ITXTPEb=2
		ELSE
		    ITXTPEb=1
		ENDIF
		MONYRLAB = .FALSE.
		IF (INDEX(LABEL,'DMY').NE.0) MONYRLAB = .TRUE.
	    ELSE
		IF(INDEX(LABEL,'MON1').NE.0)THEN
		    ITXTPEb=3
		ELSE IF(INDEX(LABEL,'MONY').NE.0)THEN
		    ITXTPEb=2
		ELSE
		    ITXTPEb=1
		ENDIF

*               Century axis mod
	        CENFLAG = .FALSE.
	        IF(INDEX(LABEL,'MULTIY').NE.0) CENFLAG = .TRUE.

*               multi-decade axis mod
	        MULTICENFLAG = .FALSE. 
	        IF(INDEX(LABEL,'MULTID').NE.0) THEN
		   CENFLAGb = .TRUE. 
		   MULTICENFLAGb = .TRUE. 
	        ENDIF
	    ENDIF
	    MONYRLABb = .FALSE.  ! Option for MON and DAY types. 
	    IF (INDEX(LABEL,'DMY').NE.0) MONYRLABb = .TRUE.
	    IF (LABEL.EQ.'HR' .OR. LABEL.EQ.'PDAY') MONYRLABb = .TRUE.
	    LABELTb=ITXTPEb*ITXLBPb
	    
	    GOTO 10
c------------------------------------------------------------------------------------

C	TXLSZE
9400	    CONTINUE
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)TCSIZEb=P(1)
	    ELSE
	       IF(IFLG1)TCSIZE=P(1)
	       IF (itflg.EQ.0.AND.IFLG1) TCSIZEb=P(1) 
	    ENDIF
	    
	    GOTO 10
C	TXLINT
9500	    CONTINUE
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)IFDBb=P(1)
	       IF(IFLG2)ISDBb=P(2)
	    ELSE
	       IF(IFLG1)IFDB=P(1)
	       IF(IFLG2)ISDB=P(2)
	       IF (itflg.EQ.0) THEN
	          IF(IFLG1)IFDBb=P(1)
	          IF(IFLG2)ISDBb=P(2)
	       ENDIF

	    ENDIF
	    GOTO 10
C	TXNMTC
9600	    CONTINUE
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)NMTCTb=P(1)
	    ELSE
	       IF(IFLG1)NMTCT=P(1)
	       IF (itflg.EQ.0 .AND.IFLG1)NMTCTb=P(1) 
	    ENDIF
	    GOTO 10
C	TXLABP
9700	    CONTINUE
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)THEN
		  ITXLBPb=P(1)
		  LABELTb=ITXTPE*ITXLBP
	       ENDIF
	    ELSE
	       IF(IFLG1)THEN
		  ITXLBP=P(1)
		  LABELT=ITXTPE*ITXLBP
		  IF (itflg.EQ.0) THEN
 		     ITXLBPb=P(1)
		     LABELTb=ITXTPE*ITXLBP
		  ENDIF
	       ENDIF
	    ENDIF
	    GOTO 10
C restore delta-T for underlay plots
C	TAXUND
9750	    CONTINUE
	    DT_SAVE = P(1)
	    GOTO 10
C	AUTOLAB
9800	    IGTLAB=0
	    IF(LABON)IGTLAB=1
	    GOTO 10
C	EVAR
9900	    CONTINUE
         print *, 'PPLUS EVAR or EPICVAR command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)
	    GOTO 10
C	DEBUG
9910	    CONTINUE

         print *, 'PPLUS DEBUG command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG

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
	    CALL PUTVAL(SYM,VIEWX,6,IER)
	    WRITE(SYM,999)'Y'
	    CALL PUTVAL(SYM,VIEWY,6,IER)
	    WRITE(SYM,999)'Z'
	    CALL PUTVAL(SYM,VIEWZ,6,IER)
	    GOTO 10
C	CLSPLT
*           Close a metafile
9950	    CALL PLTYPE(ICODE)
            IF ( save_on_exit .AND. imgname_sent ) THEN
                CALL FGD_SAVE_WINDOW(meta_file)
            ENDIF
            save_on_exit = .FALSE.
            imgname_sent = .FALSE.
            vpt_was_set = .FALSE.
	    GOTO 10
C	RENAMPL
9960	    CONTINUE

	print *, 'PPLUS RENAMPL command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG

	    GOTO 10
C	SHRINKY
9970	    CONTINUE
*	Whether to shrink ylabels automatically so they will fit in margin.
*       see YAXIS1
            SHRINKY = .FALSE.      
            IF (P(1) .EQ. 1) SHRINKY = .TRUE.      
	    GOTO 10

C	RIBFAST (RIBBON/FAST)  
9980	    CONTINUE
*	Plot style for ribbon plots: 
*	 FAST for one color per point, default is interpolate
*	 colors between points
            RIBBON_FAST = .FALSE.      
            IF (P(1) .EQ. 1) RIBBON_FAST = .TRUE.      
	    GOTO 10

C	RIBMISS (RIBBON/MISS=)  
9982	    CONTINUE
* Plot style for missing-data in ribbon plots: 
* Default: thin black line. Can request /missing=dash, /missing=blank, 
* /missing= color-palette (e.g. gray_light). The color palette 
* will typically be a one-color plot, or if a multicolor palette 
* is given its first color will be used.

            CALL RIBBON_MISSING      
	    GOTO 10


C	GAPLOC (PLOT/VS/GAPLOC)  
9984	    CONTINUE
* Extra variable sent on PLOT/VS plots, with 1 or zero to indicate
* end of segments
	    gap_var = 0
            gap_var = P(1) 
	    GOTO 10


C     CLRKLAB COLOR-KEY LABELS
9985	    CALL CLRKLAB (LABEL)
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
	GOTO 6001
C     PLOT
1500	MAKEP='PLOT'
	GOTO 6002
C     Ribbon PLOT
1550	MAKEP='RIBPLOT'
	GOTO 6002
C     READSEQ & RD
1600	CONTINUE

	print *, 'PPLUS READSEQ or RD command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

C     LIST
2900	CALL LIST(LTTOUT,pplmem,pplmem_nsize,LABEL)
	GOTO 10
3302	IF(.NOT.QUIETF)WRITE(LTTOUT,'('' EOF'')')
	SYM='*PPL$EOF'
	CALL PUTSYM(SYM,YES,3,IER)
	GOTO 10
3300	CONTINUE ! unused
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
3500	CONTINUE

	print *, 'PPLUS ENTER command '//icom(:TM_LENSTR1(icom))
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

	GOTO 10
C     MULTPLT
4900	CONTINUE
cc 4900	CALL MULT

	print *, 'PPLUS MULTPLT command '//icom(:TM_LENSTR1(icom))

         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)

	GOTO 10
C     PLOTV
5300	MAKEP='PLOTV'
5301	IF(IFLG(1).EQ.1)VANG=P(1)
	GOTO 6002
C	PLOTUV
5700	MAKEP='PLOTUV'
	GOTO 5301
6002	CONTINUE

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
	    CALL PLTIT(pplmem,pplmem_nsize)
	ELSE
	    CALL MPLOT(pplmem,pplmem_nsize)
	    IF(IMCNT.LT.MNX*MNY)GOTO 10
            rewind(lmult)
	    DO 6003 IMCNT=1,MNX*MNY
		READ(LMULT,END=6004)IPTS,MXBOT,MYLEF
		READ(LMULT,END=6004)XLABS,YLABS,RLABS,LTYPE,XLLABS,
     *		YLLABS,LLTYPE,RHLABS,USRLBS
		CALL GET(LMULT)
		IF(JTYPE.EQ.1)THEN
		    CALL GETXY(LMULT,pplmem,pplmem_nsize)
		ELSE
		    CALL GETZ(LMULT,pplmem,NX,NY)
		ENDIF
		IF(IZTYP.NE.0)CALL GETZG(LMULT)
		CALL PLTIT(pplmem,pplmem_nsize)
6003	    CONTINUE

6004	    CLOSE(LMULT)

c
c	show mult file is closed by clearing the file name
c
	    MULTFL=' '
	    CALL MPLOT2(pplmem,pplmem_nsize)
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
C       FLOWLINE (VECTOR/FLOWLINE)
9390	MAKEP='FLOW'
	IF(IFLG1)DENSITY=P(1)
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
	IF(N.GT.0)THEN
           VFRMT=LABEL
c Check if LABEL is a valid format: 
c do a test write to the string err_msg, using label as a format
           slen = TM_LENSTR(LABEL)
           IF (slen .GT. 1) THEN 
              IF (TM_HAS_STRING( label, '(I')) THEN
                 WRITE(err_msg,label,ERR=1999) 0
              ELSE
                 WRITE(err_msg,label,ERR=1999) 0.0
              ENDIF
           ENDIF
        ENDIF
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
            CALL PLTIT(pplmem,pplmem_nsize)                                               
        ELSE                                                                  
            CALL MPLOT(pplmem,pplmem_nsize)                                               
            IF(IMCNT.LT.MNX*MNY)GOTO 10                                       
            OPEN(LMULT,FILE=MULTFL,STATUS='OLD',                              
     *      FORM='UNFORMATTED')                                               
            DO 9151 IMCNT=1,MNX*MNY                                           
            READ(LMULT,END=9152)IPTS,MXBOT,MYLEF                              
	    READ(LMULT,END=9152)XLABS,YLABS,RLABS,LTYPE,XLLABS,YLLABS,LLTYPE, 
     *	    RHLABS,USRLBS                                                     
            CALL GET(LMULT)                                                   
            IF(JTYPE.EQ.1)THEN                                                
                CALL GETXY(LMULT,pplmem,pplmem_nsize)                                     
            ELSE                                                              
                CALL GETZ(LMULT,pplmem,NX,NY)                                      
            ENDIF                                                             
            IF(IZTYP.NE.0)CALL GETZG(LMULT)                                   
            CALL PLTIT(pplmem,pplmem_nsize)                                               
9151        CONTINUE    
9152        CLOSE(LMULT)
 
cc#ifdef FORTRAN_90                                           
cc9152        CLOSE(LMULT,STATUS='DELETE')                             c
cc#else
cc9152        CLOSE(LMULT,DISP='DELETE')                             
cc#endif
c                                                                  
c       show mult file is closed by clearing the file name         
c                                                                  
            MULTFL=' '                                             
            CALL MPLOT2(pplmem,pplmem_nsize)                                   
        ENDIF                                                      
        GOTO 10                                                    
C       SHAKEY                                                     
9160    call shade_key (-1)
        goto 10                                                    
C       SHASET                                                     
9170    call shade_set                                             
        goto 10
                                                 
C       S_KEY   (continuous color key, center labels, horizontal)                                                   
9165    ICODE=IFIX(P(1))
        IF (ICODE .LE. 1) call shade_key_cont (ICODE)
        call shade_key (ICODE)
        goto 10       
                                
C       SHADE_CELL   (use cellarray call for shade)                                                   
9166    ICODE=IFIX(P(1))
        call shade_cell (ICODE)
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

 9183	makep='FILLPOL'   ! makep='POLYGON'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001

 9174	call pattern_set
	goto 10

C  calendar
 9175	if(n.eq.0)then
	   set_name='GREGORIAN'
	else
	   set_name=label
	endif

	call tm_set_current_calendar (set_name, cal_status)
	if (cal_status .ne. merr_ok) GOTO 2999
	goto 10

9180    continue
        goto 10                                                    
9190	call aline (pplmem,pplmem_nsize)
	goto 10
c COLOR
9191    call set_one_color
        goto 10
c LINECOLORS
9192    call set_num_linecolors
        goto 10
c sqfill
 9380   if(labon)sqfflg=.true.
        if(VALOFF .or. n.eq.0)sqfflg=.false.
        goto 10
9999	RETURN

c Format errors
1999    slen = TM_LENSTR1(LABEL)
        err_msg = 'Invalid format specification: '//LABEL(1:slen)
        CALL errmsg (ferr_out_of_range, status, err_msg, *9999)

c Calendar errors
2999    slen = lnblk (set_name,16)
	WRITE (err_msg, 9176) set_name(1:slen)
 9176	FORMAT('Calendar "',a, '" is invalid.')
        CALL errmsg (ferr_unknown_arg, status, err_msg, *9999)

	END

