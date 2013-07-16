	SUBROUTINE CLEAR_WINDOW( wid )

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
* clear full window

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   4/2/90
* V230: 11/19/92 - clear protected colors
* V411:  9/5/95 - explicitly delete each segment before calling GCLRWK
*	to avoid XGKS memory leak in that routine
* V602: *acm* 1/07 Changes for metafile-batch mode
* PyFr *kms*  1/12 "metafile" no longer closed since it is not a separate entity
* PyFr *kms*  7/13 No need to delete segments when clearing the whole window
*                  in PyFerret; FGD_GCLRWK deletes them all in PyFerret

	include 'plot_setup.parm'
	include 'xplot_state.cmn'		! viewport info
	include	'pltcom_dat.decl'	! with gksopn
	include	'PLTCOM.DAT'	! with gksopn

* calling argument declarations:
	INTEGER	wid, ivp, iseg, istate

* internal variable declarations:

        CALL FGD_GQOPS (istate)
*  mark segment as deleted
	DO 200 ivp =  mvp_dflt, max_viewport
	   vp_seg0(ivp) = 0
	   vp_segn(ivp) = 0
 200	CONTINUE

* tell GKS to clear the named "workstation" (window)

        CALL FGD_GQOPS (istate)
	IF (istate .GE. 2) CALL FGD_GCLRWK(wid,0)

* tell PPLUS to release protected colors (on viewports and overlays)
        CALL PPL_SHASET( 'RESET' )

	RETURN
	END
