	SUBROUTINE CLEAR_VP

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
	LOGICAL ITS_BATCH_GRAPHICS

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
