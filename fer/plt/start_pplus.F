	SUBROUTINE START_PPLUS

* initialize Denbo's PPLUS program which is imbedded within this program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/29/87 
* revision 0.1 - 6/3/87  - added color definitions
* revision 0.2 - 7/28/87 - added INQUIRE to see if wkstn. has colors
* revision 1.0 - 6/29/88 - removed color definitions to DEFINE_COLORS
*			   removed mode determination to INITIALIZE
* revision 1.1 - 8/11/88 - added GKS deferral state call
* revision 1.2 - 8/16/88 - added GKS_WINDOWS to allow FERRET on BETA
*			   removed GKS deferral state call
* revision 1.3 - 8/26/88 - added PPL SIZE (9,10) to initialize
*			   since the SIZE command is removed from DISP_INIT
* revision 1.4 - 9/8/88  - initialize PPLUS with PPL_in_FERRET
*			 - add EXTERNALS to force loads from FERRET library
*			   instead of PPLUS library
* revision 1.5 - 4/1/89  - transfer MODE WAIT directly to PPLUS variable deflts
* V200:  12/6/89 - GKS state determined by mode_GKS, color setup removed
*	 9/19/90 - warning message when GKS mode is cancelled
* Ultrix/RISC port - commented out GKSFORBND.DEF include: not used
* V300:  3/11/93 - sets a default color spectrum
* V301:	 11/3/93 - calls SET_VIEWPORT to define VP_WIDTH and VP_height symbols
*	 1/24/94 - turn on metafiles in DISP_INIT instead of here
*	           (resolve problem of metafiles created with wrong name)
* V450:  TEMPORARY EXPERIMENTAL change to test metafile-only output
* V450:  3/4/97 - Mod to support metafile-only (no X) output through a
*		cuckoo's egg implantation of meta_wsid in place of wsid

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xgks_devices.cmn'
	include	'tmap_pplv11inc/system_inc.decl'	! with echof
	include	'pplv11inc/SYSTEM.INC'	! with echof
	include	'tmap_pplv11inc/plt_inc.decl'	! with icode
	include	'pplv11inc/PLT.INC'	! with icode
	include	'tmap_pplv11inc/gkscm1_inc.decl'	! with wsid
	include	'pplv11inc/GKSCM1.INC'	! with wsid
	include	'tmap_pplv11inc/switch_inc.decl'	! with deflts
	include	'pplv11inc/SWITCH.INC'	! with deflts
	include	'tmap_pplv11inc/pltcom_dat.decl'	! with gksopn
	include	'pplv11inc/PLTCOM.DAT'	! with gksopn
	include	'tmap_pplv11inc/ppl_in_ferret.cmn'	! with ppl_in_ferret
!	include	'sys_library/gksdefs.bnd'		! gks parameters
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XGKS_DEVICES.CMN'
	INCLUDE	'TMAP_PPLV11INC:SYSTEM_INC.DECL'	! with echof
	INCLUDE	'PPLV11INC:SYSTEM.INC'	! with echof
	INCLUDE	'TMAP_PPLV11INC:PLT_INC.DECL/LIST'	! with icode
	INCLUDE	'PPLV11INC:PLT.INC/LIST'	! with icode
	INCLUDE	'TMAP_PPLV11INC:GKSCM1_INC.DECL/LIST'	! with wsid
	INCLUDE	'PPLV11INC:GKSCM1.INC/LIST'	! with wsid
	INCLUDE	'TMAP_PPLV11INC:SWITCH_INC.DECL/LIST'	! with deflts
	INCLUDE	'PPLV11INC:SWITCH.INC/LIST'	! with deflts
	INCLUDE	'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'	! with gksopn
	INCLUDE	'PPLV11INC:PLTCOM.DAT/LIST'	! with gksopn
	INCLUDE	'PPLV11INC:PPL_in_FERRET.CMN'	! with PPL_in_FERRET
!	INCLUDE	'SYS$LIBRARY:GKSDEFS.BND'		! GKS parameters
#endif
	
* internal parameter declarations:
	CHARACTER	dummy_filename*1
	PARAMETER     ( dummy_filename	  = ' ' )

* internal variable declarations
	LOGICAL		ITS_BATCH_GRAPHICS, batch

* external declarations to force loading of modules used by PPLUS
	EXTERNAL	GET_VP_FROM_FERRET, CLEAR_VP, INTERRUPT,
     .			SEG_ON, SEG_OFF

* still need to cold-start graphics ?
	IF ( pplus_started ) RETURN

* are we in no-X mode?
	batch = ITS_BATCH_GRAPHICS()

* tell PPLUS that it is under FERRET control without viewports
	PPL_in_FERRET = .TRUE.
	PPL_viewports_active = .FALSE.

* set PPLUS /WAIT vs /NOWAIT mode directly from FERRET mode
	deflts(5) = mode_wait	! sets swait default

* tailor output graphics device to environment
	IF ( mode_gks ) THEN
!	   icode = 3		! PLTYPE 3 (GKS)
	   icode = 1		! PLTYPE 3 WS=... is done below
	ELSEIF ( .NOT.interactive ) THEN
	   icode = 0		! PLTYPE 0 (PPLUS metafile)
	ELSE
	   icode = 1		! PLTYPE 1 (Tek terminal)
	   CALL WARN('MODE GKS is disabled.')
	   CALL WARN('Some graphics functionality will not be available.')
	ENDIF

* open PLOT5
	CALL OPNPPL ( dummy_filename,41,42,43,44,ttout_lun,45,46,47,48 )
	echof = .FALSE.
	pplus_started = .TRUE.

* define plot size
	CALL COLOR(1)
	CALL DISP_RESET

* set correct plot device (e.g. "PLTYPE 3 WS=VSII" )
	IF ( batch ) THEN
	   IF ( mode_gks ) CALL SEND_PLTYPE(.TRUE.)
	ELSE
	   IF ( mode_gks ) CALL SEND_PLTYPE(.FALSE.) ! 1/94 was "mode_metafile"
	ENDIF

	IF ( .NOT.gksopn ) RETURN	! flag was set by PPLUS

* start GKS on FERRET side
	wn_open( wsid )   = .TRUE.
	wn_active( wsid ) = .TRUE.
!	wn_open( 1 )   = .TRUE.	! note that wsid only HAPPENS TO BE the
!	wn_active( 1 ) = .TRUE. ! transformation number

!	CALL DEFINE_COLORS( wsid )

* set up the default spectrum
        CALL PPL_SHASET( 'SPECTRUM=default' )

* set up axis lengths and PLOT+ symbols
	CALL SET_VIEWPORT( mvp_dflt )

* set GKS to regenerate whenever workstation viewport is modified - avoids
*  bug interaction with segments when the first plot is done directly via PPLUS
*  note: subsequent windows will still have their deferral states at default
*	 but no problems are caused by that because FERRET and PPLUS agree on
*	 the window viewport coordinates (see PPLUS SIZE2)
!	CALL GSDS( wsid, gasap, gallow )
! further bug - repeated SET WINDOW/SIZE=x for same x cause segment to go
! invisible
	RETURN
	END
