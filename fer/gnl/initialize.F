	SUBROUTINE INITIALIZE

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
* initialize run-time variables and I/O in preparation to begin execution
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/26/86
* revision 0.1 - 12/11/86 - explicit OPEN of SYS_OUTPUT with 255 byte RECL
*			    and 255 byte RECL on journal file OPEN
* revision 0.2 - 1/30/87 - modified for mr_protected declared as INTEGER
*			   and RADIUS made a parameter in FERRET.PARM
* revision 0.3 - 5/1/87  - EXTERNAL *CONTROL and no LIST file opened
* revision 0.4 - 9/14/87 - added VMS calls to turn on line recall
* revision 0.5 - 11/30/87- added call to enable ^C interrupts
* revision 0.6 - 2/18/88 - converted X axes to "regular" format
* revision 0.7 - 5/2/88  - changed GFDL.JNL to GFDL_JOURNAL (logical) and added
*			   ERR= branch to OPEN
* revision 0.8 - 6/29/88  - changed GFDL_JOURNAL to FERRET_JOURNAL
*			    and today_time in hours:minutes, only
*			    determine mode (interactive) here
* revision 0.9 - 7/21/88 - memory initialization done by INIT_MEMORY
* revision 1.0 - 8/12/88 - added FERRET_INIT optioal initialization file
* revision 1.1 -11/22/88 - new TM IO library
* revision 1.2 - 2/2/89  - allow program to continue w/out journal file
* V200:  6/22/89 - define special grids at run time
*	 12/6/89 - initialize mode_GKS and mode_wait
* Unix/RISC port - 1/91 ... several OS dependencies fixed
*                - 9/91 - fixed err handling when journal file cant be opened
*                - 10/91 - hook to enable PLOT+ braindead features
*                - 10/91 -   FER_INIT ==> .ferret
* V230:  5/14/92 - on SGI (with no backing store) set REFRESH mode
* V300:  5/12/93 - split journal file init into another file
*		 - relocated sgi IFDEF to default REFRESH to ON
*       10/25/93 - default MODE REFRESH ON for **all** systems
*	 2/24/94 - small change due to mode_state as a 2D variable
*	 3/14/94 - base initial state of mode refresh on "ifdef X_REFRESH"
* V312:  5/19/94 - significant portions of routine removed to new MAIN program
* V313: 10/27/94 - *sh* AIX port: avoid the routine name "time"
* V420:  9/19/95 - call TM_INIT_DYN_GRIDS for dynamic (implicit) grid creation
* Linux port *kob* 3/96 - had to add calls to use F90 data and time routine
*		     rather than f77 default routines.

#ifdef unix
        include 'tmap_format/tmap_errors.parm'  ! error codes
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xio.cmn_text'
	external xio_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/gfdl_vms.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/gfdl.parm'			! from phil/sieg
	include	'ferret_cmn/xonedim.cmn'		! from phil.sieg
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xtoday.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'  ! error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:GFDL.PARM'			! from Phil/Sieg
	INCLUDE	'FERRET_CMN:XONEDIM.CMN'		! from Phil.Sieg
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XTODAY.CMN'
	INCLUDE '($SSDEF)'				! LIB_ return status
	INCLUDE '($JPIDEF)'				! for VMS mode code

	INTEGER	SMG$CREATE_VIRTUAL_KEYBOARD, SMG$CREATE_KEY_TABLE,
     .		LIB$SYS_TRNLOG
	INTEGER	vax_code, vms_mode
#endif

* local variable declarations:
	LOGICAL	GKS_WINDOWS, fexist
	INTEGER	TM_LENSTR1, ito, ifrom, mem_blk, mr, status
	REAL	radian, cm2deg
	CHARACTER TM_INQ_PATH*128

*************** I/O ************************************************************
# ifdef unix
	interactive = .TRUE.
#else
* determine VMS mode (BATCH, etc.)
	CALL GET_VMS_MODE( vms_mode )
	interactive = vms_mode .EQ. JPI_K_INTERACTIVE 
#endif

* determine default GKS usage based on system logon device
	mode_GKS = GKS_WINDOWS()
	mode_wait = .NOT.mode_GKS	! no wait after plot on window device

#ifdef unix

* turn on segment-based screen refresh or screen updates dont always work
#ifdef X_REFRESH
      CALL TM_SET_FREE_EVENT(1)
      mode_state( pmode_xwindows, 1 ) = .TRUE.
#else
      mode_state( pmode_xwindows, 1 ) = .FALSE.
#endif
        CONTINUE
#else
* console output with long records
	OPEN (	FILE='SYS_OUTPUT:',
     .		UNIT=ttout_lun,
     .		RECL = 255,
     .		STATUS='OLD' )
* listing file - by default uses ttout log. unit

* special VMS set-up for last 20 line recall
	vax_code = smg_create_virtual_keyboard ( vms_kbd_id )
	IF ( vax_code .NE. vms_smg_success ) WRITE ( ttout_lun, * )
     .		'err-->smg_create_virt_key:',vax_code
	vax_code = smg_create_key_table ( vms_kbd_tbl )
	IF ( vax_code .NE. vms_smg_success ) WRITE ( ttout_lun, * )
     .		'err-->smg_create_key_tabl:',vax_code

#endif

* set up special grids
	CALL DEFINE_SPECIAL_GRIDS   ! label 1000 moved here 9/91 *sh*

* initialize dynamic grid management
	CALL TM_INIT_DYN_GRIDS

*************** TODAY'S DATE ***************************************************


#ifdef AIX_XLF
	CALL AIX_DATE( today_date )
	CALL AIX_TIME( today_time )
#elif F90_DATE_TIME
	CALL FDATE(today_date)
	CALL FTIME(today_time)
#else
	CALL DATE( today_date )
	CALL TIME( today_time )
#endif
	today_time = today_time(:5)	! w/out seconds

*************** STATE VARIABLES ************************************************
* see BLOCK DATA initialization of COMMON/ XPROG_STATE /

*********************** GRIDS **************************************************
	CALL GFDL_TERMS

	RADIAN=57.29578							! 1027
	cm2deg = radian / radius
	
!	ito = line_subsc1(mpsxt)
!	DO 210 ifrom = 1, line_dim(mpsxt)
!	   line_mem ( ito ) = xt ( ifrom )		! point location
!	   ito = ito + 1
! 210	CONTINUE
!	DO 212 ifrom = 1, line_dim(mpsxt)
!	   line_mem ( ito ) = dxt( ifrom ) * cm2deg	! grid box size
!	   ito = ito + 1
! 212	CONTINUE
!
!	ito = line_subsc1(mpsxu)
!	DO 220 ifrom = 1, line_dim(mpsxu)
!	   line_mem ( ito ) = xv ( ifrom )
!	   ito = ito + 1
! 220	CONTINUE
!	DO 222 ifrom = 1, line_dim(mpsxu)
!	   line_mem ( ito ) = dxu( ifrom ) * cm2deg
!	   ito = ito + 1
! 222	CONTINUE
!
!	ito = line_subsc1(mpsyt)
!	line_start(mpsyt) = yt( 1 )
!	DO 230 ifrom = 1, line_dim(mpsyt)
!	   line_mem ( ito ) = yt ( ifrom )
!	   ito = ito + 1
! 230	CONTINUE
!	DO 232 ifrom = 1, line_dim(mpsyt)
!	   line_mem ( ito ) = dyt( ifrom ) * cm2deg
!	   ito = ito + 1
! 232	CONTINUE
!
!	ito = line_subsc1(mpsyu)
!	line_start(mpsyu) = yv( 1 )
!	DO 240 ifrom = 1, line_dim(mpsyu)
!	   line_mem ( ito ) = yv ( ifrom )
!	   ito = ito + 1
! 240	CONTINUE
!	DO 242 ifrom = 1, line_dim(mpsyu)
!	   line_mem ( ito ) = dyu( ifrom ) * cm2deg
!	   ito = ito + 1
! 242	CONTINUE
!
!	ito = line_subsc1(mpszt)
!	line_start(mpszt) = zdzz( 1 ) * .01
!	DO 250 ifrom = 1, line_dim(mpszt)
!	   line_mem ( ito ) = zdzz( ifrom ) * .01
!	   ito = ito + 1
! 250	CONTINUE
!	DO 252 ifrom = 1, line_dim(mpszt)
!	   line_mem ( ito ) = dz  ( ifrom ) * .01
!	   ito = ito + 1
! 252	CONTINUE
!
!	ito = line_subsc1(mpszw)
!	line_start(mpszw) = zdz( 1 ) * .01
!	DO 260 ifrom = 1, line_dim(mpszw)
!	   line_mem ( ito ) = zdz( ifrom     ) * .01
!	   ito = ito + 1
! 260	CONTINUE
!	DO 262 ifrom = 1, line_dim(mpszw)
!	   line_mem ( ito ) = dzz( ifrom + 1 ) * .01 ! Phil/Sieg start at 0
!	   ito = ito + 1
! 262	CONTINUE

*************** MISCELLANEOUS **************************************************
* TMAP librabry routine errors to same output unit as other errors
	lunit_errors = err_lun

*************** BRAINDEAD PLOT+ ******************************************
        CALL PPL_BDEAD

	RETURN

	END

