	SUBROUTINE CLEAR_VP

* clear full window or single viewport of it depending on plot state variables
* (this routine to hide FERRET common from PPLUS)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/20/89
* V200:  3/22/90 - check if WSID NE 0 before clearing wkstn (for CANC WIND/ALL)
* V411:  9/5/95 - use CLEAR_WINDOW to clear the full screen
*	(to avoid the XGKS memory leak from GCLRWK)
*        9/9/97 - if batch is active, just clear workstation since XGKS
*        doesn't delete metafile segments, and it avoids gdelseg warnings

#ifdef unix
	include 'ferret_cmn/xplot_state.cmn'		! viewport info
	include 'tmap_pplv11inc/gkscm1_inc.decl'	! with wsid
	include 'pplv11inc/GKSCM1.INC'	! with wsid
#else
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'		! viewport info
	INCLUDE 'TMAP_PPLV11INC:GKSCM1_INC.DECL'	! with WSID
	INCLUDE 'PPLV11INC:GKSCM1.INC'	! with WSID
#endif

* internal variable declarations:
	INTEGER	ivp, iseg

	IF ( vp_num .EQ. mvp_dflt ) THEN

*  clear entire screen
	   IF ( wsid .NE. 0 ) THEN
* explicitly delete segments to avoid XGKS memory leak
	      DO 110 ivp =  mvp_dflt, max_viewport
	         IF ( .not. ITS_BATCH_GRAPHICS() .and.
     1  	      vp_seg0(ivp) .NE. 0 ) THEN 
	            DO 100 iseg = vp_seg0(ivp)+1 , vp_segn(ivp)
 100	            CALL GDSG( iseg ) 
	         ENDIF
*  and mark segment as deleted
	         vp_seg0(ivp) = 0
	         vp_segn(ivp) = 0
 110	      CONTINUE

	      CALL GCLRWK(WSID,0)
	   ENDIF

	ELSE

* clear viewport only - delete segments
	   DO 200 iseg = vp_seg0(vp_num)+1 , vp_segn(vp_num)
 200	   CALL GDSG( iseg ) 
* immediate update of screen to reflect deleted segments
	   CALL GUWK( wsid, 1 )
	ENDIF

	vp_seg0( vp_num ) = curr_seg_name

	RETURN
	END
