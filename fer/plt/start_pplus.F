	SUBROUTINE START_PPLUS

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
* V610 *acm* 3/08 - add new MODE NLEVELS; Store the setting for the default
*                   number of contour/fill levels.


	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xgks_devices.cmn'
	include	'system_inc.decl'	! with echof
	include	'SYSTEM.INC'	! with echof
	include	'plt_inc.decl'	! with icode
	include	'PLT.INC'	! with icode
	include	'gkscm1_inc.decl'	! with wsid
	include	'GKSCM1.INC'	! with wsid
	include	'switch_inc.decl'	! with deflts
	include	'SWITCH.INC'	! with deflts
	include	'pltcom_dat.decl'	! with gksopn
	include	'PLTCOM.DAT'	! with gksopn
	include	'ppl_in_ferret.cmn'	! with ppl_in_ferret
!	include	'gksdefs.bnd'		! gks parameters
        INCLUDE 'lunits_inc.decl'   ! std err unit number
	INCLUDE 'LUNITS.INC'
	
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

* Give PPLUS the unit number for std error, to write error messages

        lerror = err_lun

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

* Set default (initial) number of color levels

        CALL SET_DEFAULT_NLEVELS( mode_arg(pmode_nlevels, 1) )

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
