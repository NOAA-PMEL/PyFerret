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
*        1.30.99 *js* Eliminated change of 9/9/97, since it caused memory
*          leak and we don't seem to get warnings anymore.
* V602:  1/2007 *acm* When writing metafiles in batch mode, we do want to
*        skip the call to GDSG. The segments are in fact deleted by gclosews 
*        (gclwk called in pltit when closing one plot and opening a new one)
*        and that results in the error message when gdsg is called here.
* V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* PyFr *kms*  7/13 No need to delete segments when clearing the whole window
*                  in PyFerret; FGD_GCLRWK deletes them all in PyFerret
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.

	include 'xplot_state.cmn'		! viewport info
	include 'gkscm1_inc.decl'	! with wsid
	include 'GKSCM1.INC'	! with wsid

* internal variable declarations:
	INTEGER	ivp, iseg

	IF ( vp_num .EQ. mvp_dflt ) THEN

*  clear entire screen
	   IF ( wsid .NE. 0 ) THEN
	      DO 110 ivp =  mvp_dflt, max_viewport
*  and mark segment as deleted
	         vp_seg0(ivp) = 0
	         vp_segn(ivp) = 0
 110	      CONTINUE

	      CALL FGD_GCLRWK(WSID,0)
	   ENDIF

	ELSE

* clear viewport only - delete segments
	   DO 200 iseg = vp_seg0(vp_num)+1 , vp_segn(vp_num)
 	      CALL FGD_GDSG( iseg ) 
 200	   CONTINUE
* immediate update of screen to reflect deleted segments
	   CALL FGD_GUWK( wsid, 1 )
	ENDIF

	vp_seg0( vp_num ) = curr_seg_name

	RETURN
	END
