	SUBROUTINE CLEAR_WINDOW( wid )

* clear full window and close metafile

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   4/2/90
* V230: 11/19/92 - clear protected colors
* V411:  9/5/95 - explicitly delete each segment before calling GCLRWK
*	to avoid XGKS memory leak in that routine

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xplot_state.cmn'		! viewport info
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'	
#endif

* calling argument declarations:
	INTEGER	wid, ivp, iseg

* internal variable declarations:

! --- start of 9/95 addition
* explicitly delete segments to avoid XGKS memory leak
	DO 200 ivp =  mvp_dflt, max_viewport
	   IF ( vp_seg0(ivp) .NE. 0 ) THEN 
	      DO 100 iseg = vp_seg0(ivp)+1 , vp_segn(ivp)
 100	      CALL GDSG( iseg ) 
	   ENDIF
*  and mark segment as deleted
	   vp_seg0(ivp) = 0
	   vp_segn(ivp) = 0
 200	CONTINUE
! --- end of 9/95 addition

* tell GKS to clear the named "workstation" (window)
	CALL GCLRWK(wid,0)

* tell PPLUS to close the metafile if it's open
	CALL PPLCMD ( from, line, 0, 'CLSPLT', 1, 1 )

* tell PPLUS to release protected colors (on viewports and overlays)
        CALL PPL_SHASET( 'RESET' )

	RETURN
	END
