      SUBROUTINE SWAP_ACTIVE_WINDOW()

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
* v5.22 *kob* - routine to swap and unswap the active
*               window id's for double buffered windows/animation. 
*               While animating, the inactive window is the one the 
*               animation is being copied to.  The active window is the
*               unmapped window in which the plots/shades are being drawn.

      INTEGER wsid, anim_id

#ifdef unix
	include 'ferret.parm'
	include 'xprog_state.cmn'
	include 'xinterrupt.cmn'
	include 'plot_setup.parm'
 10     include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
        include 'plt_inc.decl'     ! with width/height
	include	'pltcom_dat.decl' 
	include	'PLTCOM.DAT'         	  ! with asize,bsize
        include 'plt_inc.decl'     ! with width/height
        include 'PLT.INC'
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	                ! with gks common

#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif
        REAL  xln, yln, dmaxx, dmaxy, fctx, fcty
        INTEGER ier, id, itype, imeter, irx, iry,i, idim,
     .          iseg


*    calculate size of the window open
*     - taken from set_window.F

*        CALL SYMBEL(0,0,0,.01,20,' ')
        IF(ASIZE.GT.BSIZE)THEN
           FCTY=BSIZE/ASIZE
           FCTX=1.0
        ELSE
           FCTX=ASIZE/BSIZE
           FCTY=1.0
        ENDIF            

        CALL GQWKC(wsid,IER,ID,ITYPE)
        CALL QUERY_DISPLAY_SIZE(ITYPE,IER,IMETER,
     .       DMAXX,DMAXY,IRX,IRY)

        XLN=ASIZE*2.54E-5       ! Denbo milli-inches to meters
        YLN=BSIZE*2.54E-5           
*     re-scale as requested
        xln = wn_scale(wsid) * xln
        yln = wn_scale(wsid) * yln
        wkxmn = MIN( xlowlef(wsid), dmaxx-xln )
        wkymn = MIN( ylowlef(wsid), dmaxy-yln )
        WKXMX=WKXMN+XLN
        WKYMX=WKYMN+YLN

* Check if animation is being turned on, or if we're done
        IF (animate .EQ. .TRUE.) THEN 
* Find the currently active window
           DO wsid = 1,max_windows
              IF (wn_active(wsid) ) GOTO 50
           end do
 50        wn_scale(win_id9) = wn_scale(wsid)

*     swap id's
           active_win_id = wsid
           anim_id = active_win_id
           wsid = win_id9
*     deactivate what will become the animated window
           CALL CLEAR_WINDOW( anim_id )
           CALL GDAWK( anim_id)
           wn_active(anim_id) = .FALSE.
*     activate the unmapped window
           CALL GACWK( wsid)
           wn_active(wsid) = .TRUE.
           CALL CLEAR_WINDOW( wsid )
           
*     if wkxms and wkymx are not zero, then we've already done a plot
*     so set window size accordingly
           IF (WKXMX .NE. 0 .AND. WKYMX .NE. 0) THEN
*     set its correct size if we've done a plot
              CALL GSWKWN(wsid,0.0,FCTX,0.0,FCTY)
              CALL SET_WINDOW_SIZE (wsid,WKXMN,WKXMX,WKYMN,WKYMX)
*              CALL GSWKWN(anim_id,0.0,FCTX,0.0,FCTY)
*              CALL SET_WINDOW_SIZE (anim_id,WKXMN,WKXMX,WKYMN,WKYMX)
           ELSE
              CALL GSELNT(wsid)
              CALL GSELNT(anim_id)
           ENDIF

        CALL XGKS_X_EVENTS

        ELSE 
           CALL CLEAR_WINDOW( wsid )
           IF ( vp_num .EQ. mvp_dflt ) THEN
*     return to full screen - clear away everything
              DO 100 i = mvp_dflt, max_viewport
                 IF (iseg .gt. 0) THEN
                    vp_seg0(iseg) = 0
                    vp_segn(iseg) = 0
                 ENDIF
 100          CONTINUE
           ELSE
*     delete any segments showing in this viewport
              DO 200 iseg = vp_seg0(vp_num)+1, vp_segn(vp_num)
*             CALL GDSG( iseg ) 
                 IF (iseg .gt. 0) THEN
                    vp_seg0(iseg) = 0
                    vp_segn(iseg) = 0
                 ENDIF
 200          CONTINUE
              CALL GUWK( wsid, 1 )
              CALL PPL_SHASET( 'PROTECT' ) ! protect the colors already on-screen
           ENDIF
           
*     done animating so switch id's back
           wsid = active_win_id
           anim_id = win_id9
*     deactivate what is now the unmapped window
           CALL GDAWK(anim_id)
           wn_active(anim_id) = .FALSE.
*     close unmapped window
           CALL GCLWK(anim_id)
           wn_open ( anim_id ) = .FALSE.
*     activate the mapped window for continued ferret use
           CALL GACWK(wsid)
           wn_active(wsid) = .TRUE.
*     kludge to get window in proper state for next plot - otherwise 
*     it doesn't clear upon next plotting commmand
           CALL SYMBEL(0,0,0,.01,20,' ')
*     set up workstation and window sizes
           CALL GSWKWN(wsid,0.0,FCTX,0.0,FCTY)
           CALL SET_WINDOW_SIZE (wsid,WKXMN,WKXMX,WKYMN,WKYMX)
             
              
           ENDIF
           
           RETURN
           END
      
