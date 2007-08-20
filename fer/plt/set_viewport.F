      SUBROUTINE SET_VIEWPORT( ivp )

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
* set up the output viewport within the current window

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  8/31/88
* revision 0.1 -  9/29/88 - allow separate GKS normaliz. trans. for each window
* V200: 11/30/89 - pass viewport flag to PPLUS
*	  4/2/90 - clear screen with CLEAR_WINDOW
* Unix/RISC port - 2/91 - special cases for GKSDEFS include file
* V230:  11/4/92 - added call to SET_AX_SIZES to auto-size viewport axes
*        11/9/92 - incorporated SHASET PROTECT to save color tables
*        3/29/93 - restore the old-style functioning of DEFINE VIEW/SIZE
* V540: 9/01 *sh* - DEF VIEW/AXES: viewports based on *axis* positions
* V580: 11/04 *acm* New symbols for viewport scaling
* V602: *acm* 1/07 Changes for metafile-batch mode

	include 'ferret.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
	include	'pltcom_dat.decl' ! with sizing data, etc.
	include	'PLTCOM.DAT'	         ! with sizing data, etc.
        include 'plt_inc.decl'     ! with width/height
        include 'PLT.INC'
	include	'ppl_in_ferret.cmn'	! with ppl_in_ferret
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	! with gks common
!	include 'gksdefs.bnd'		! with gclip
C  clipping indicator (lines from VMS SYS$LIBRARY:GKSDEFS.BND)
	INTEGER*4  GNCLIP,GCLIP
	DATA       GNCLIP,GCLIP/  0,    1/
	
* calling argument declarations:
	INTEGER	ivp

* internal variable declarations:
	LOGICAL clip, newstyle_vp
	INTEGER	iseg, i, old_vp, istate
	REAL	scale, fctx, fcty, xclip, yclip, xfrac, yfrac,
     .          xcei, ycei, xoei, yoei, xcv, xov, ycv, yov

* main window opened yet ?
	IF ( .NOT.pplus_started ) CALL START_PPLUS

* only works under GKS
	IF ( .NOT. gksopn ) RETURN

* make this the new viewport
	old_vp = vp_num
	vp_num = ivp

* emulate the obsolete /SIZE qualifier?
	newstyle_vp = vp_size(ivp) .GT. 0.0

	IF ( old_vp .EQ. mvp_dflt .OR. vp_num .EQ. mvp_dflt ) THEN
* return to full screen - clear away everything
	   CALL CLEAR_WINDOW( wsid )
	   DO 100 i = mvp_dflt, max_viewport
	      vp_seg0(i) = 0
	      vp_segn(i) = 0
 100	   CONTINUE
	ELSE
* set new viewport - delete any segments showing in this viewport
	   DO 200 iseg = vp_seg0(vp_num)+1, vp_segn(vp_num)
 200	   CALL GDSG( iseg ) 
           CALL PPL_SHASET( 'PROTECT' ) ! protect the colors already on-screen
	ENDIF

* indicate where new segments in this viewport of this window begin
	vp_seg0( vp_num ) = curr_seg_name
	vp_segn( vp_num ) = curr_seg_name

* pass viewport flag to PPLUS
	PPL_viewports_active = vp_num .NE. mvp_dflt

* For viewport/AXES viewports, update ASIZE and BSIZE here,
* or else the first viewport plotted is in the wrong place.
        IF (vp_by_axis(vp_num)) THEN
           CALL GQOPS (istate)  ! In batch mode the graphics may not be active now. 
           IF (istate .GE. 2) CALL SIZE (width,height)
        ENDIF

* NDC space goes 0 to fctx and 0 to fcty (corresponding to PLOT+ SIZE command)
	IF(ASIZE.GT.BSIZE)THEN
	   FCTY=BSIZE/ASIZE
	   FCTX=1.0
	ELSE
	   FCTX=ASIZE/BSIZE
	   FCTY=1.0
	ENDIF

* decode clipping
	clip = vp_xclip(vp_num) .NE. unspecified_val4
	IF ( clip ) THEN
	   xclip = vp_xclip( vp_num )
	   yclip = vp_yclip( vp_num )
	ELSE
	   xclip = 1.0
	   yclip = 1.0
	ENDIF

* determine the scale factor to be applied to the axis length to create
* the "text_prominence" effect
	scale = 1.0 / SQRT( ABS(vp_size(vp_num)) )

* ... V5.22 default behaviors (viewport not specified by axis)
        xoei = xwhitelo  ! ppl xorigin
        yoei = ywhitelo  ! ppl yorigin
        xcei = xwhitehi  ! white space to right of X axis
        ycei = ywhitehi  ! white space for docmtn at top of plot
        xov = vp_xorg(vp_num)
        xcv = xclip
        yov = vp_yorg(vp_num)
        ycv = yclip
        IF (vp_by_axis(vp_num)) THEN
          CALL VP_AXLIM2LIM( scale, xov,  yov,  xcv,  ycv,
     .                              xov,  yov,  xcv,  ycv,
     .                              xoei, yoei, xcei, ycei )
        ENDIF
* set viewport in NDC to limits requested

	CALL GSVP(wsid,fctx*xov, fctx*xcv,
     .		       fcty*yov, fcty*ycv )

* set world coordinate window to preserve 1:1 aspect

	CALL GSWN(wsid, 0.0, ( xcv-xov )*asize*scale,
     .		        0.0, ( ycv-yov )*bsize*scale  )

	IF ( newstyle_vp ) THEN

* auto-size the axes to fit the plot window
           xfrac = scale*(xcv-xov)
           yfrac = scale*(ycv-yov)

           CALL SET_AX_SIZES( xfrac, yfrac,
     .                        xoei, yoei, xcei, ycei )

* Set more symbols describing the viewport

           CALL SET_VP_SYMBOLS (scale, vp_size(vp_num), xfrac, yfrac,
     .			        xoei, yoei, xcei, ycei,
     .                          xov,  yov,  xcv,  ycv)

	   CALL GSELNT(wsid)     ! no clipping 11/92 (below) for PPL BOX ON 

	ELSE
* emulate old-style, non-automated viewports
* select normalization transformation and turn on clipping
	   CALL GSELNT(wsid)
	   IF ( clip ) THEN
	      CALL GSCLIP(gclip)
	   ELSE
	      CALL GSCLIP(gnclip)
	   ENDIF
	ENDIF

* success
	RETURN
	END
