      SUBROUTINE CANCEL_WINDOW( status )

* delete a window from the display.  If it was currently active then make the
* main window active

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/29/88
* V200:	12/20/89 - changed from UIS to Xwindows
*	 3/13/90 - start PPLUS if not already started, check mode_GKS
* V320:  2/28/95 - set a flag for ferret_dispatch so it can keep GUI informed
*		   of window opens/closes
* V400:  6/20/95 - remove changes from 2/28/95 -- state change on windows
*		will be detected by mismatch between wn_open AND 
*		gui_window_reported

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xgui.cmn'
	include	'tmap_pplv11inc/pltcom_dat.decl'	! with sizing data, etc.
	include	'pplv11inc/PLTCOM.DAT'	! with sizing data, etc.
	include	'tmap_pplv11inc/gkscm1_inc.decl'	! with gks common
	include	'pplv11inc/GKSCM1.INC'	! with gks common
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
	INCLUDE	'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'	! with sizing data, etc.
	INCLUDE	'PPLV11INC:PLTCOM.DAT/LIST'	! with sizing data, etc.
	INCLUDE	'TMAP_PPLV11INC:GKSCM1_INC.DECL/LIST'	! with GKS common
	INCLUDE	'PPLV11INC:GKSCM1.INC/LIST'	! with GKS common
#endif
	
* calling argument declarations:
	INTEGER		status

* internal variable declarations:
	LOGICAL 	TM_DIGIT
	INTEGER 	n, n_lo, n_hi, lo
	CHARACTER	buffer*24

* local parameter declarations:
	INTEGER		slash_all
	PARAMETER     ( slash_all = 0+1 )

* main window opened yet ?
	CALL START_PPLUS

* windows only from GKS
	IF ( .NOT.mode_gks ) THEN
	   CALL WARN('Windows are only available with GKS output')
	   CALL WARN('See HELP on SET MODE GKS')
	   GOTO 1000
	ENDIF

* get window numbers to close
	IF ( qual_given( slash_all ) .GT. 0 ) THEN
* ... /ALL
	   n_lo = 1
	   n_hi = max_windows
	
	ELSEIF ( num_args .EQ. 0 ) THEN
* ... CANCEL WINDOW -- what window ?
	   CALL ERRMSG
     .		(ferr_invalid_command,status,'specify window # or /ALL',*5000)

	ELSE
* ... use number given
	   buffer = cmnd_buff(arg_start(1):arg_end(1)) 
	   IF ( .NOT.TM_DIGIT( buffer ) ) GOTO 5020
	   READ ( buffer, *, ERR=5020 ) n
	   n_lo = n
	   n_hi = n
	ENDIF	   

* valid number ?
 60	IF ( n_lo.LT.1 .OR. n_hi.GT.max_windows ) CALL ERRMSG
     .		( ferr_out_of_range, status, 'legal numbers are 2-8', *5000 )

* loop through all requested windows
	DO 100 n = n_lo, n_hi
	   IF ( .NOT.wn_open( n ) ) GOTO 100

* currently active ?
	   IF ( wn_active(n) ) THEN
* ... clear segments from it
	      CALL SET_VIEWPORT( mvp_dflt )
* ... deactivate it
	      CALL GDAWK( n )
	      wn_active( n ) = .FALSE.
* ... activate lowest available window, if any
	      DO 70 lo = 1, max_windows
	         IF ( lo.NE.n  .AND. wn_open(lo) ) THEN
	            CALL GACWK( lo )
	            wn_active( lo ) = .TRUE.
	            wsid = lo
	            GOTO 80
	         ENDIF
 70	      CONTINUE
	      wsid = 0		! no active window, now
	   ENDIF
	      
* close the specified window
 80	   CALL GCLWK( n )
! old UIS code
!	   CLOSE ( UNIT=n+50, ERR=5010 )
	   wn_open ( n ) = .FALSE.
	   wn_scale( n ) = wn_dflt_scale

* and set flag so that the GUI will be informed when this window is reopened 
!	   gui_window_reported(n) = .FALSE.   ! No - done in ferret_dispatch

 100	CONTINUE

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit(s)
 5000	RETURN
! 5010	CALL ERRMSG(ferr_erreq,status,'closing window',*5000)
 5020	CALL ERRMSG(ferr_invalid_command,status,buffer,*5000)
	END
