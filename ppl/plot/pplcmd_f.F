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
* 11/2022 *acm* Code cleanup: Remove and issue note for long-obsolete PPLUS commands
* 12/2022 *acm* Code cleanup: Major rewrite to replace the computed GOTO statement 
*                             "GOTO (100,200,300,500 ...), I"  with IF blocks

c	fromi	command file or device (/dev/tty or TT: for terminal)
c	linei	input line for parameters
c	isi	length of line in characters
c	combuf() command buffer for executing commands from memory
c	icmdim	dimensioned length of combuf
c	icmsze	number of lines in combuf
c
        include 'COMEPL.INC'  ! added 9.6.88 j davison 
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
	include 'data_inc.decl'
	include 'DATA.INC'
	include 'cmrd_inc.decl'
	include 'CMRD.INC'
	include 'cmrdl_inc.decl'
	include 'CMRDL.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
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
	LOGICAL old_comm
	LOGICAL IFLG1,IFLG2,LABON,VALOFF,TBOTH
	DIMENSION X1(5),Y1(5),Y2(4),X3(2),IX3(5)
	DIMENSION X4(4)
	INTEGER	  num_comm
	PARAMETER(num_comm = 126)
	CHARACTER COMM(num_comm)*7
	EQUIVALENCE (DT,X3(1)),(ITYPET,IX3(1)),(TSMX,X4(1))
	EQUIVALENCE (XLO,X1(1)),(YLO,Y1(1)), (HLAB1,Y2(1))
	INTEGER	ISTAT,IFLAG 
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
 
	CHARACTER*15  TMPCHR
	character*2048 set_name

	external scale
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

      makep = ' '

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
	    M=IBFDIM
	    M=MIN0(M,BUFSZE)
	    DO I=1,M
	      CMDBUF(I)=COMBUF(I)
	    ENDDO
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

C	If GKS WS is active, flush output buffer before leaving if just plotted
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

C   Is this a known command?

	ICOM7(1:7)=ICOM(1:7)
	DO I=1,num_comm
	   IF(ICOM7.EQ.COMM(I)) GOTO 21
	ENDDO

	IF(ICOM(1:1).EQ.'%')THEN

*  TMAP check that output window is active
           IF ( PPL_in_FERRET ) CALL CK_GKS_ACTIVE( *9999 ) 
*  TMAP addition to turn on segment
           IF ( PPL_in_FERRET ) CALL SEG_ON( )

	   CALL PRMTIV(pplmem,pplmem_nsize)
           IF ( PPL_in_FERRET ) CALL SEG_OFF
	ELSE
	    CALL SPWN
	ENDIF

	GOTO 10

C known commands
21	CONTINUE
	IFLG1=(M.GE.1.AND.IFLG(1).NE.0)
	IFLG2=(M.GE.2.AND.IFLG(2).NE.0)
	LABON=(LABEL.EQ.'ON'.OR.LABEL.EQ.'on')
	TBOTH=(LABEL.EQ.'BO'.OR.LABEL.EQ.'bo')
	IF (TBOTH) LABON = .TRUE.
	VALOFF=(INDEX(LABEL,'OFF')+INDEX(LABEL,'off')).NE.0

C Obsolete commands. Issue an error message
      old_comm = .FALSE.
      IF (ICOM7 .EQ. 'AUTO') old_comm = .TRUE. 
      IF (ICOM7 .EQ. 'BAUD') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RWDS' ) old_comm = .TRUE.
      IF (ICOM7 .EQ. 'READS') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'VARS') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'CLINE') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'SKPS') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RWD') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RD') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'SKP') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'TKTYPE') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'ENTER') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'SAVE') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'GET') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'PARAM') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'MULTPLT') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RDCOM') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'LINFIT') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'GRID') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'TEKNME') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'AUTOLAB') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'ECHO') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'EVAR') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'DEBUG') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RESET') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'TMDEBUG') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'PIXMAP') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'RENAMPL') old_comm = .TRUE.
      IF (ICOM7 .EQ. 'USR') old_comm = .TRUE.


      IF (old_comm) THEN
         err_msg = 'PPLUS Command not supported '//icom(:TM_LENSTR1(icom))
         CALL ERRMSG (ferr_not_implemented, status, err_msg, *9999)
         GOTO 10
      ENDIF

	IF (ICOM7.EQ.'TMDEBUG') GOTO 10
	IF (ICOM7.EQ.'BAUD') RETURN

C PLTYPE
      IF (ICOM7 .EQ. 'PLTYPE') THEN
	   ICODE=IFIX(P(1))
        IF (ICODE .NE. 3) THEN
           IF(.NOT.QUIETF)WRITE(LTTOUT,'(1X,A)')
     .          'PLTYPE 3 ONLY IS SUPPORTED IN THIS VERSION OF PPLUS'
           GOTO 10
        ENDIF
	CALL PLTYPE(ICODE)
	GOTO 10
      ENDIF

C SIZE
      IF (ICOM7 .EQ. 'SIZE') THEN
	WIDTH=P(1)
	HEIGHT=P(2)
	SYM='*PPL$WIDTH'
	CALL PUTVAL(SYM,WIDTH,6,IER)
	SYM='*PPL$HEIGHT'
	CALL PUTVAL(SYM,HEIGHT,6,IER)
	GOTO 10
      ENDIF

C SMOOTH
      IF (ICOM7 .EQ. 'SMOOTH') THEN
	IF(M.EQ.0)GOTO 10
	CALL SMOOTH(pplmem,NX,NY,NX,NY,IFIX(P(1)))
	GOTO 10
      ENDIF

C XLAB
      IF (ICOM7 .EQ. 'XLAB') THEN
	XLAB=LABEL
	GOTO 10
      ENDIF

C YLAB
      IF (ICOM7 .EQ. 'YLAB') THEN
	YLAB=LABEL
	GOTO 10
      ENDIF

C FORMAT
      IF (ICOM7 .EQ. 'FORMAT') THEN
	IFORM=LABEL
	CALL UPPER(IFORM,80)
	SYM='*PPL$FORMAT'
	ISYM=LNBLK(IFORM,80)
	CALL PUTSYM(SYM,IFORM,ISYM,IER)
	LSCTD=SCTD
	GOTO 10
      ENDIF

C XAXIS
      IF (ICOM7 .EQ. 'XAXIS') THEN
	DO I=1,M
	    IF(I.GE.4) EXIT
	    IF(IFLG(I).NE.0)X1(I)=P(I)
	ENDDO
	IF(M.GT.0)THEN
	    IAUTOX=0
	    X1(3)=SIGN(ABS(X1(3)),X1(2)-X1(1))
	ELSE
	    IAUTOX=1
	ENDIF
	GOTO 10
      ENDIF

C YAXIS
      IF (ICOM7 .EQ. 'YAXIS') THEN
	DO I=1,M
	    IF(M.GE.4)EXIT
	    IF(IFLG(I).NE.0)Y1(I)=P(I)
	ENDDO
	IF(M.GT.0)THEN
	    IAUTOY=0
	    Y1(3)=SIGN(ABS(Y1(3)),Y1(2)-Y1(1))
	ELSE
	    IAUTOY=1
	ENDIF
	GOTO 10
      ENDIF
      
C NLINES
      IF (ICOM7 .EQ. 'NLINES') THEN
	CALL RSMNMX
	LINEN=0
	IBASE=1
	PBUF=0
	LNUM=1
	GOTO 10
      ENDIF

C LINE
      IF (ICOM7 .EQ. 'LINE') THEN
	I=IFIX(P(1))
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
      ENDIF

C LABS
      IF (ICOM7 .EQ. 'LABS') THEN
	IF(P(1).EQ.0..OR.M.EQ.0)THEN
	    DO I=1,NLABS
		LLTYPE(I)=0
		RHLABS(I)=-1.0
	    ENDDO
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
      ENDIF

C LABSET
      IF (ICOM7 .EQ. 'LABSET') THEN
	DO I=1,M
	    IF(IFLG(I).NE.0)Y2(I)=P(I)
	ENDDO
	GOTO 10
      ENDIF

C LEVELS
      IF (ICOM7 .EQ. 'LEVELS') THEN
	CALL LEVEL
	GOTO 10
      ENDIF

C CONSET
      IF (ICOM7 .EQ. 'CONSET') THEN
	IF(IFLG1)HGT=P(1)
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
      ENDIF

C CROSS
      IF (ICOM7 .EQ. 'CROSS') THEN
	ICROSS=0
	IF(LABON)ICROSS=1
	IF(IFLG(1).EQ.1)ICROSS=ICROSS+INT(P(1))
	GOTO 10
      ENDIF

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

C XFOR
      IF (ICOM7 .EQ. 'XFOR') THEN
	i = STR_UPCASE (TMPCHR, LABEL(1:5))
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

      ENDIF

C YFOR
      IF (ICOM7 .EQ. 'YFOR') THEN
	i = STR_UPCASE (TMPCHR, LABEL(1:15))
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
	
      ENDIF

C XVALOFF
      IF (ICOM7 .EQ. 'XVALOFF') THEN
	XVALOFF=P(1)
	i = XVALOFF
	IF (FLOAT(i) .NE. XVALOFF) THEN
           err_msg = 'Offset must be an integer'
           CALL ERRMSG (ferr_out_of_range, status, err_msg, *9999)
	ENDIF
	GOTO 10
      ENDIF

C YVALOFF
      IF (ICOM7 .EQ. 'YVALOFF') THEN
	i = YVALOFF
	IF (FLOAT(i) .NE. YVALOFF) THEN
           err_msg = 'Offset must be an integer'
           CALL ERRMSG (ferr_out_of_range, status, err_msg, *9999)
	ENDIF
	GOTO 10
      ENDIF

C ORIGIN
      IF (ICOM7 .EQ. 'ORIGIN') THEN
	IF(IFLG1)XORG=P(1)
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
      ENDIF

C WINDOW
      IF (ICOM7 .EQ. 'WINDOW') THEN
	IWIND=0
	IF(LABON) IWIND=1
	GOTO 10
      ENDIF

C METRIC
      IF (ICOM7 .EQ. 'METRIC') THEN
	CALL METRIC
	GOTO 10
      ENDIF

C ENGLISH
      IF (ICOM7 .EQ. 'ENGLSH') THEN
	CALL ENGLSH
	GOTO 10
      ENDIF

C LIMITS
      IF (ICOM7 .EQ. 'LIMITS') THEN
	CALL UPPER(LABEL,2048)
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
      ENDIF

C TAXIS
      IF (ICOM7 .EQ. 'TAXIS') THEN
	ITFLG=0
	IF(LABON)THEN
	    IF(SYAXIS)THEN
		ITFLG=-1
	ELSE
		ITFLG=1
	    ENDIF
	ENDIF
	IF (TBOTH) ITFLG=2
	DO I=1,M
	    IF(I.GE.3)GOTO 5402
	    IF(IFLG(I).NE.0)X3(I)=P(I)
	ENDDO
	GOTO 10
5402	DO J=I,M
	    IF(IFLG(J).NE.0)IX3(J-2)=IFIX(P(J))
	ENDDO
	GOTO 10
      ENDIF
      
C TMIN
      IF (ICOM7 .EQ. 'TMIN') THEN
	IF(N.NE.0)THEN
	    IT1=LABEL(2:11)
	    IF(ITSTRT.EQ.' ')ITSTRT=IT1
	ENDIF

	TLO=1.-DIFF(IT1,ITSTRT)/DT 
	GOTO 10
      ENDIF

C TMAX
      IF (ICOM7 .EQ. 'TMAX') THEN
	IF(N.NE.0)THEN
	    IT2=LABEL(2:11)
	ENDIF
	GOTO 10
      ENDIF

C TSTART
      IF (ICOM7 .EQ. 'TSTART') THEN
 	IF (IFORM(1:3).EQ.'EPI' .OR. IFORM(1:3).EQ.'BIB') THEN
	    GOTO 10
	ELSE
	    IF(N.EQ.0)LABEL(2:)=IT1
	    ITSTRT=LABEL(2:11)
	    TLO=1.-DIFF(IT1,ITSTRT)/DT
	    GOTO 10
	ENDIF
      ENDIF

C TICS
      IF (ICOM7 .EQ. 'TICS') THEN
	DO I=1,M
	    IF(I.GT.4)EXIT
	    IF(IFLG(I).NE.0)X4(I)=P(I)
	ENDDO
	IF(M.GE.5.AND.IFLG(5).EQ.1)ITX=P(5)
	IF(M.GE.6.AND.IFLG(6).EQ.1)ITY=P(6)
* The following lets the user plot small tics with PPL TICS
	MULTICENFLAG = .FALSE.
	GOTO 10

      ENDIF 

C PEN
      IF (ICOM7 .EQ. 'PEN') THEN
	I=IFIX(P(1))
	if (label .ne. ' ') then 
	   call upnsquish (label,strr,end)
	   if (strr .eq. 'SAVE') then
	      ipen_save(i) = ipen(i)
	      goto 10
	   elseif (strr .eq. 'RESTORE') then
	      ipen(i) = ipen_save(i)
	      goto 10
	   endif
	endif
	IF(M.LT.2.OR.IFLG(2).EQ.0)GOTO 10
	IF(I.LT.1)I=0
	IPEN(I)=IFIX(P(2))
	GOTO 10
      ENDIF

C AXLEN
      IF (ICOM7 .EQ. 'AXLEN') THEN
	IF(IFLG1)XLEN=P(1)
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
      ENDIF

C AXLABP
      IF (ICOM7 .EQ. 'AXLABP') THEN
	IF(IFLG1)LABELX=IFIX(P(1))
	IF(IFLG2)LABELY=IFIX(P(2))
	GOTO 10
      ENDIF

C AXLINT
      IF (ICOM7 .EQ. 'AXLINT') THEN
	IF(IFLG1)LINTX=IFIX(P(1))
	IF(IFLG2)LINTY=IFIX(P(2))
	GOTO 10
      ENDIF

C AXTYPE
      IF (ICOM7 .EQ. 'AXTYPE') THEN
	IF(IFLG1)ITYPEX=IFIX(P(1))
	IF(IFLG2)ITYPEY=IFIX(P(2))
	GOTO 10
      ENDIF

C AXLSZE
      IF (ICOM7 .EQ. 'AXLSZE') THEN
	IF(IFLG1)XCSIZE=P(1)
	IF(IFLG2)YCSIZE=P(2)
	GOTO 10
      ENDIF

C AXNSIG
      IF (ICOM7 .EQ. 'AXNSIG') THEN
	IF(IFLG1)NSIGX=IFIX(P(1))
	IF(IFLG2)NSIGY=IFIX(P(2))
	GOTO 10
      ENDIF

C AXNMTC
      IF (ICOM7 .EQ. 'AXNMTC') THEN
	IF(IFLG1)NMTCX=IFIX(P(1))
	IF(IFLG2)NMTCY=IFIX(P(2))
	GOTO 10
      ENDIF

C AXATIC
      IF (ICOM7 .EQ. 'AXATIC') THEN
	IF(IFLG1)NTICX=IFIX(P(1))
	IF(IFLG2)NTICY=IFIX(P(2))
	GOTO 10
      ENDIF

C ROTATE
      IF (ICOM7 .EQ. 'ROTATE') THEN
	IF(LABON)THEN
	    CALL ROTATE(90.)
	ELSE
	    CALL ROTATE(0.)
	ENDIF
	GOTO 10
      ENDIF

C COMMENT COMMAND 'C'
      IF (ICOM7 .EQ. 'C') THEN
	GOTO 10
      ENDIF

C BOX
      IF (ICOM7 .EQ. 'BOX') THEN
	IP(1)=0
	IF(LABON)IP(1)=1
	GOTO 10
      ENDIF

C RLABS
      IF (ICOM7 .EQ. 'RLABS') THEN
	I=IFIX(P(1))
	IF(IFLG2)THEN
	    RLABS(I)=P(2)
	ELSE
	    RLABS(I)=0.
	ENDIF
	GO TO 10
      ENDIF

C AXSET
      IF (ICOM7 .EQ. 'AXSET') THEN
	DO I=1,M
	    IF(IFLG(I).NE.0) IAXON(I)=P(I)
	ENDDO
	GO TO 10
      ENDIF

C 'BLANK COMMAND'
      IF (ICOM7 .EQ. ' ') THEN
	GO TO 10
      ENDIF

C LLABS
      IF (ICOM7 .EQ. 'LLABS') THEN
	IF(.NOT.IFLG1)GOTO 10
	I=P(1)
	IF(IFLG2)THEN
	   XLLABS(I)=P(2)
	   YLLABS(I)=P(3)
	ELSE
	   LLTYPE(I)=0
	ENDIF
	IF(M.GE.4.AND.IFLG(4).NE.0)LLTYPE(I)=P(4)
	GOTO 10
      ENDIF

C LEV
      IF (ICOM7 .EQ. 'LEV') THEN
	CALL LEV
	GOTO 10
      ENDIF

C DATPT
      IF (ICOM7 .EQ. 'DATPT') THEN
	IF(IFLG1)IZTYP=P(1)
	IF(IFLG2)IZMRK=P(2)
	GOTO 10
      ENDIF

C PLTNME
      IF (ICOM7 .EQ. 'PLTNME') THEN

            IF ( save_on_exit .AND. imgname_sent ) THEN
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

      ENDIF

C HLABS
      IF (ICOM7 .EQ. 'HLABS') THEN
	    IF(.NOT.IFLG1)GOTO 10
	    I=P(1)
	    IF(IFLG2)THEN
		RHLABS(I)=P(2)
	    ELSE
		RHLABS(I)=-1.0
	    ENDIF
	    GOTO 10
      ENDIF

C CONPRE
      IF (ICOM7 .EQ. 'CONPRE') THEN
	    IF(N.EQ.0)THEN
		CONPRE=' '
	    ELSE
		CONPRE=LABEL
	    ENDIF
	    GOTO 10
      ENDIF

C CONPST
      IF (ICOM7 .EQ. 'CONPST') THEN
	    IF(N.EQ.0)THEN
		CONPST=' '
	    ELSE
		CONPST=LABEL
	    ENDIF
	    GOTO 10
      ENDIF

C TRANSXY
      IF (ICOM7 .EQ. 'TRANSXY') THEN
	    I=IFIX(P(1))
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
      ENDIF

C MARKH
      IF (ICOM7 .EQ. 'MARKH') THEN
	    I=IFIX(P(1))
	    IF(IFLG2)HMARK(I)=P(2)
	    IF(M.EQ.1)HMARK(I)=0.08
	    IF(M.GE.3)nskpsym(I) = p(3)
	    GOTO 10
      ENDIF

C DFLTFNT
      IF (ICOM7 .EQ. 'DFLTFNT') THEN
	    IF(N.EQ.0)GOTO 10
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
      ENDIF

C TIME
      IF (ICOM7 .EQ. 'TIME') THEN

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
	   
c--------------------------------------------------------------------

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

c------- end of Time definition for second time axis 

	ENDIF  ! end of  IF (ICOM7 .EQ. 'TIME')     


C TXTYPE
      IF (ICOM7 .EQ. 'TXTYPE') THEN	

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
      ENDIF
c------------------------------------------------------------------------------------

C TXLSZE
      IF (ICOM7 .EQ. 'TXLSZE') THEN
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)TCSIZEb=P(1)
	    ELSE
	       IF(IFLG1)TCSIZE=P(1)
	       IF (itflg.EQ.0.AND.IFLG1) TCSIZEb=P(1) 
	    ENDIF
	    
	    GOTO 10
      ENDIF

C TXLINT
      IF (ICOM7 .EQ. 'TXLINT') THEN
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
      ENDIF

C TXNMTC
      IF (ICOM7 .EQ. 'TXNMTC') THEN
	    IF (itflg .EQ. 2) THEN
	       IF(IFLG1)NMTCTb=P(1)
	    ELSE
	       IF(IFLG1)NMTCT=P(1)
	       IF (itflg.EQ.0 .AND.IFLG1)NMTCTb=P(1) 
	    ENDIF
	    GOTO 10
      ENDIF

C TXLABP
      IF (ICOM7 .EQ. 'TXLABP') THEN
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
	    
      ENDIF


C restore delta-T for underlay plots
C TAXUND
      IF (ICOM7 .EQ. 'TAXUND') THEN
	    DT_SAVE = P(1)
	    GOTO 10
      ENDIF

C TITLE
      IF (ICOM7 .EQ. 'TITLE') THEN
	    IF(N.EQ.0)THEN
		LAB1=' '
	    ELSE
		LAB1=LABEL
	    ENDIF
	    IF(IFLG1)HLAB1=P(1)
	    GOTO 10
      ENDIF

C VPOINT
      IF (ICOM7 .EQ. 'VPOINT') THEN
	    IF(IFLG1)VIEWX=P(1)
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
      ENDIF

C CLSPLT
      IF (ICOM7 .EQ. 'CLSPLT') THEN
*           Close a metafile, cancel viewports
	    CALL PLTYPE(ICODE)
            IF ( save_on_exit .AND. imgname_sent ) THEN
                CALL FGD_SAVE_WINDOW(meta_file)
            ENDIF
            save_on_exit = .FALSE.
            imgname_sent = .FALSE.
            vpt_was_set = .FALSE.
	    GOTO 10
      ENDIF

C SHRINKY
      IF (ICOM7 .EQ. 'SHRINKY') THEN
*	Whether to shrink ylabels automatically so they will fit in margin.
*       see YAXIS1
            SHRINKY = .FALSE.      
            IF (P(1) .EQ. 1) SHRINKY = .TRUE.      
	    GOTO 10
      ENDIF

C RIBFAST (RIBBON/FAST)  
      IF (ICOM7 .EQ. 'RIBFAST') THEN
*	Plot style for ribbon plots: 
*	 FAST for one color per point, default is interpolate
*	 colors between points
            RIBBON_FAST = .FALSE.      
            IF (P(1) .EQ. 1) RIBBON_FAST = .TRUE.      
	    GOTO 10
      ENDIF

C RIBMISS (RIBBON/MISS=) 
      IF (ICOM7 .EQ. 'RIBMISS') THEN
* Plot style for missing-data in ribbon plots: 
* Default: thin black line. Can request /missing=dash, /missing=blank, 
* /missing= color-palette (e.g. gray_light). The color palette 
* will typically be a one-color plot, or if a multicolor palette 
* is given its first color will be used.

            CALL RIBBON_MISSING      
	    GOTO 10
      ENDIF

C GAPLOC (PLOT/VS/GAPLOC)  
      IF (ICOM7 .EQ. 'GAPLOC') THEN
* Extra variable sent on PLOT/VS plots, with 1 or zero to indicate
* end of segments
	    gap_var = 0
            gap_var = P(1) 
	    GOTO 10
      ENDIF

C CLRKLAB COLOR-KEY LABELS
      IF (ICOM7 .EQ. 'CLRKLAB') THEN
	    CALL CLRKLAB (LABEL)
	    GOTO 10
      ENDIF

C CONTOUR
      IF (ICOM7 .EQ. 'CONTOUR') THEN
	MAKEP='CONTOUR'
	if(iflg1)then
	    ivcomp=p(1)
	else
	    ivcomp=1
	endif
	if(iflg2)cangle=p(2)
	GOTO 6001
      ENDIF


C PLOT
      IF (ICOM7 .EQ. 'PLOT') THEN
	MAKEP='PLOT'
	GOTO 6002
      ENDIF

C Ribbon PLOT
      IF (ICOM7 .EQ. 'RIBBON') THEN
	MAKEP='RIBPLOT'
	GOTO 6002
      ENDIF

C LIST
      IF (ICOM7 .EQ. 'LIST') THEN
	CALL LIST(LTTOUT,pplmem,pplmem_nsize,LABEL)
	GOTO 10
      ENDIF

C VIEW
      IF (ICOM7 .EQ. 'VIEW') THEN
	MAKEP='VIEW'
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
	GOTO 6001
      ENDIF

C PLOTV
      IF (ICOM7 .EQ. 'PLOTV') THEN
	MAKEP='PLOTV'
	IF(IFLG(1).EQ.1)VANG=P(1)
	GOTO 6002
      ENDIF

C PLOTUV
      IF (ICOM7 .EQ. 'PLOTUV') THEN
	MAKEP='PLOTUV'
	IF(IFLG(1).EQ.1)VANG=P(1)
	GOTO 6002
      ENDIF


C VELVCT
      IF (ICOM7 .EQ. 'VELVCT') THEN
	MAKEP='VELVCT'
	IF(IFLG(1).EQ.1)VANG=P(1)
	GOTO 6002
      ENDIF

C VECTOR
      IF (ICOM7 .EQ. 'VECTOR') THEN
	MAKEP='VECTOR'
	IF(IFLG1)VSKPX=P(1)
	IF(IFLG2)VSKPY=P(2)
	GOTO 6001
      ENDIF

C FLOWLINE (VECTOR/FLOWLINE)
      IF (ICOM7 .EQ. 'FLOWLIN') THEN
	MAKEP='FLOW'
	IF(IFLG1)DENSITY=P(1)
	GOTO 6001
      ENDIF

C VECSET
      IF (ICOM7 .EQ. 'VECSET') THEN
	IF(IFLG1)VLEN=P(1)
	IF(IFLG2)VUSRLN=P(2)
	VAUTO=M.LT.2
	GOTO 10
      ENDIF

C VECKEY
      IF (ICOM7 .EQ. 'VECKEY') THEN
	IF(IFLG1)VXLOC=P(1)
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
      ENDIF

C SHADE
      IF (ICOM7 .EQ. 'SHADE') THEN
        MAKEP = 'SHADE'                                                       
        IF(N.NE.0)LAB1=LABEL                                                  
        ICOUNT=ICOUNT+1                                                       
        CALL PLTIT(pplmem,pplmem_nsize)
        GOTO 10  
      ENDIF
                             
C SHAKEY      
      IF (ICOM7 .EQ. 'SHAKEY') THEN                                               
	call shade_key (-1)
	goto 10  
      ENDIF
                                                  
C SHASET  
      IF (ICOM7 .EQ. 'SHASET') THEN                                                
	call shade_set                                             
        goto 10
      ENDIF

C S_KEY   (continuous color key, center labels, horizontal)                  
      IF (ICOM7 .EQ. 'S_KEY') THEN    
	ICODE=IFIX(P(1))
        IF (ICODE .LE. 1) call shade_key_cont (ICODE)
        call shade_key (ICODE)
        goto 10  
      ENDIF
                 
C SHADE_CELL   (use cellarray call for shade)                  
      IF (ICOM7 .EQ. 'S_CELL') THEN                                                     
	ICODE=IFIX(P(1))
        call shade_cell (ICODE)
        goto 10   
      ENDIF

c FILL      
      IF (ICOM7 .EQ. 'FILL') THEN               
	makep='FILL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
      ENDIF

C PLOTPOL
      IF (ICOM7 .EQ. 'PLOTPOL') THEN
 	makep='PLOTPOL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
      ENDIF

C FILLPOL
      IF (ICOM7 .EQ. 'FILLPOL' .OR. ICOM7.EQ.'POLYGON') THEN
 	makep='FILLPOL'
        if(iflg1)then
            ivcomp=p(1)
        else
            ivcomp=1
        endif
        if(iflg2)cangle=p(2)
        GOTO 6001
      ENDIF

C PATSET
      IF (ICOM7 .EQ. 'PATSET') THEN
	call pattern_set
	goto 10
      ENDIF

C CALENDAR
      IF (ICOM7 .EQ. 'CALENDA') THEN
 	if(n.eq.0)then
	   set_name='GREGORIAN'
	else
	   set_name=label
	endif

	call tm_set_current_calendar (set_name, cal_status)
	if (cal_status .ne. merr_ok) GOTO 2999
	goto 10
      ENDIF

C ALINE
      IF (ICOM7 .EQ. 'ALINE') THEN
	call aline (pplmem,pplmem_nsize)
	goto 10
      ENDIF

c COLOR
      IF (ICOM7 .EQ. 'COLOR') THEN
	call set_one_color
        goto 10
      ENDIF

c LINECOLORS
      IF (ICOM7 .EQ. 'LINECOL') THEN
	call set_num_linecolors
        goto 10
      ENDIF

c SQFILL
      IF (ICOM7 .EQ. 'SQFILL') THEN
	if(labon)sqfflg=.true.
        if(VALOFF .or. n.eq.0)sqfflg=.false.
        goto 10
      ENDIF


* Call PLTIT to draw a plot

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
	CALL PLTIT(pplmem,pplmem_nsize)
	GOTO 10

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

