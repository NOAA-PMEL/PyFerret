      SUBROUTINE SET_WINDOW( status )

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
* create a new output window and de-activate (but don't remove) the last

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 6/28/88
* revision 0.1 - 8/1/88  - set window/size defaults to current window
* revision 0.2 - 8/26/88 - added qualifiers /LOCATION
* revision 0.3 -  9/8/88 - added /ASPECT and /TEXT_PROMINENCE
* revision 0.4 - 9/29/88 - /ASPECT clears screen to avoid future redraws
*			 - separate normalization transform for each window
* V200: 12/15/89 - VAX binding required for Xwindows
*	12/20/89 - allow all windows to have been cancelled, set up bundles
*	 3/22/90 - POP the window to the top with an escape function
*	  4/2/90 - clear screen with CLEAR_WINDOW
* Unix/RISC port - 2/26/91 - commented out unused SYS$LIBRARY include statement
*                - TM_LENSTR1 was declared LOGICAL !
*                - added NEW_GKS_WINDOW for OS independence
*                - fixed /ASPECT default behavior to consistent "edge" (10/91)
* V230: 3/92: Sun port -
*                - replaced CALL GQDSP with CALL QUERY_DISPLAY_SIZE
*                  and CALL GSWKVP with CALL SET_WINDOW_SIZE for ATC GKS
*      10/92:    - update PPL_viewports_active so metafiles start fresh
*     11/4/92    - change to SET_PLOT_SIZE call for auto viewports
*         note about ABSIZE/BSIZE v.s. HEIGHT/WIDTH:  They differ in value by
*              ASIZE = WIDTH*1000  **and** asize,bsize are updated **after**
*              the next plot is produced whereas width,height are updated
*              immediately when PPL SIZE is given (in SET_PLOT_SIZE)
* V320: 8/94 *kob* IBM port- revealed that "physical" should be defined as
*			     logical but was defined as integer
* V411: 9/6/95 *sh* changes to CLEAR_WINDOW (explicit segment deleting) caused
*	SET WINDOW/CLEAR/SIZE=0.2 to crash Ferret.  Indications are that this
*	is timing-related (plot must be complex, does not happen in totalview)
*	Changes here to clear window **before** other actions.
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.
* *js* Don't create new window with set win/new if batch graphics active
*      Don't set window to any number but 1 if batch graphics active
* *kob* 1.29.97 changed "can\'t" to cannot for compiler reasons
* V540 *sh* 9/01 - pass axis white space to SET_AX_SIZE (from PLOT_SETUP.cmn)
* V600 *acm* 9/05  add SET WINDOW/TITLE.  If no title has been set, set it to
*                  the session date:time. Save window_title in xplot_setup.cmn
*                  for use in subsequent windows. Also define a symbol WIN_TITLE
*  v600 *acm* 5/2006 Flag no_plot_yet: changes so user can change the number of
*                  line colors.
* V602: *acm* 1/07 Changes for metafile-batch mode
* V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* PyFr *kms* 2/2013  Add /QUALITY=... /AALIAS /NOAALIAS /ENGINE=...
* PyFr *kms* 4/2013  remove specific reference to /AALIAS /NOAALIAS in warning message
* PyFr  *kms*  6/13  Add /COLOR= /OPACITY= to specify background color
* PyFr  *kms*  6/13  Add /THICKEN= to specify line width scaling factor
* PyFr  *kms*  6/13 Add /XPIXELS, /YPIXELS, /XINCHES, /YINCHES to SET WINDOW;
*                   Call WINDOW_SIZE_FROM_OPTS to deal with them
* PyFr  *kms*  7/13 Call SET_PPL_VALUES to set scaled lengths and label sizes;
*                   set physical_chg to .TRUE. if START_PPLUS called or
*                   if the engine changes
* V69x *sh* 21/1/15 clarify data type of arguments sent to QUERY_DISPLAY_SIZE
* V7   *kms*  10/16 Add SET WINDOW /SCALE=

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
        include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'errmsg.parm'
        include 'ppl_in_ferret.cmn' ! with PPL_viewports_active
	include	'pltcom_dat.decl'
	include	'PLTCOM.DAT'         	  ! with asize,bsize
        include 'plt_inc.decl'     ! with width/height
        include 'PLT.INC'
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	                ! with gks common
	include 'axis_inc.decl'		! axis lengths
	include 'AXIS.INC'		        ! axis lengths
        include 'xplot_setup.cmn'       ! for animate boolean
        include 'xtoday.cmn'            ! with today_date and today_time
        include 'fgrdel.cmn'            ! with windowdpix and windowdpiy

* calling argument declarations:
       INTEGER status

* internal variable declarations:
	   LOGICAL TM_DIGIT, MATCH4, FGD_ENGINE_MATCH,
     .         physical_chg, relocate, chg_window, axis_ref
       INTEGER TM_LENSTR1, TM_LENSTR, ier, id, itype, imeter, irx, iry,
     .         wn, n, qp, ivp, colon_pos, slen, tlen, i, s1,
     .         bkgcolor, xpixels, ypixels,
     .         qp_op, qp_xl, qp_yl, qp_scl
	   REAL    xln, yln, xmn, xmx, ymn, ymx, scl,
     .         xstretch, ystretch, aspect, ref_len, xsc, ysc,
     .         wmkx, wmky, scale, opac
	   REAL*4  rzero, rone, fcty, fctx,
     .         xwlo, ywlo, xwhi, ywhi, dfltthick,
     .         redfrac, grnfrac, bluefrac, opaqfrac,
     .         xinches, yinches, sc, owd,
     .         xloc, yloc, scalefrac, opacity
       CHARACTER UPPER_CASE*1, GKS_X_CONID*16, warnmsg*80,
     .         arg1*1, buffer*2048, xwind_id*80, short_buff*24

* local parameter declarations:
	INTEGER slash_new,
     .          slash_clear,
     .          slash_title,
     .          slash_quality,
     .          slash_aalias,
     .          slash_noaalias,
     .          slash_engine,
     .          slash_color,
     .          slash_opacity,
     .          slash_thicken,
     .          slash_textprom,
     .          slash_outline,
     .          slash_scale,
     .          slash_wmark,
     .          slash_wmark_scale,
     .          slash_wmark_opac,
     .          slash_wmark_xloc,
     .          slash_wmark_yloc
	PARAMETER ( slash_new      =  1+1,
     .              slash_clear       =  4+1,
     .              slash_title       =  5+1,
     .              slash_quality     =  6+1,
     .              slash_aalias      =  7+1,
     .              slash_noaalias    =  8+1,
     .              slash_engine      =  9+1,
     .              slash_color       = 10+1,
     .              slash_opacity     = 11+1,
     .              slash_thicken     = 12+1,
     .              slash_textprom    = 17+1,
     .              slash_outline     = 18+1,
     .              slash_scale       = 19+1,
     .              slash_wmark       = 20+1,
     .              slash_wmark_scale = 21+1,
     .              slash_wmark_opac  = 22+1,
     .              slash_wmark_xloc  = 23+1,
     .              slash_wmark_yloc  = 24+1 )

	rzero = 0.0
	rone = 1.0
        physical_chg = .FALSE.

* /TITLE=text
* This text becomes the name of the workstation. The name gets truncated
* at the first blank space. Replace any internal spaces with underscore.

	IF ( .NOT.pplus_started ) window_title = ' '

        tlen = 1
	qp = qual_given( slash_title )
	IF (qp .GT. 0 .AND. .NOT. animate) THEN
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   window_title, status )
           tlen = TM_LENSTR1(window_title)
           DO 70 i = 1, tlen
             IF (window_title(i:i) .EQ. ' ') window_title(i:i) = '_'
 70        CONTINUE

* By default let the session date and time be incorporated into the name.

        ELSE IF (qp .EQ. 0 .AND. window_title .EQ. ' ') THEN
           s1 = 1
           IF (today_date(1:1) .EQ. ' ') s1 = 2
           tlen = TM_LENSTR1(today_date)
           window_title = today_date(s1:tlen)//':'//today_time

* Or the directory.
!  This works, but with slashes the path name does not get appended.
!  gks calls shorten FERRET_2_/home/ja8/ansley/FERRET/fer
!  to just the directory name.  If we do use the directory, change
! the code in open_gks_ws.F to do it the same way.

c           call getenv ('PWD',window_title)
c           tlen = TM_LENSTR1(window_title)
c           DO 75 i=1, tlen
c              IF (window_title(i:i) .EQ. '/') window_title(i:i) = '_'
c 75        CONTINUE

        ENDIF
        tlen = TM_LENSTR1(window_title)
        ier = 0
        CALL SETSYM ("WIN_TITLE "//window_title(:tlen), tlen+10, ier, status)

* main window opened yet ?
	IF ( .NOT.pplus_started ) THEN
	   CALL START_PPLUS(.TRUE.)
*          this opens a window, so need to set physical_chg .TRUE. to resize it
           physical_chg = .TRUE.
	ENDIF

* are we in a valid GKS window environment ?
	IF ( .NOT. gksopn	) RETURN

* get number
	IF ( num_args .EQ. 0 .AND. qual_given(slash_new) .GT. 0 ) THEN

* ... find next free number
* *kob* 1/00 - check to make sure user doesnt open window 9 - reserved
*              for animation purposes
           chg_window = .TRUE.
           DO 50 n = 1, (max_windows-1)
              IF ( .NOT.wn_open( n ) ) GOTO 60
 50        CONTINUE
           CALL ERRMSG(ferr_prog_limit,status,'# windows',*5000)
	ELSEIF ( num_args .GT. 0 ) THEN
* ... use number given
	   buffer = cmnd_buff(arg_start(1):arg_end(1))
	   IF ( .NOT.TM_DIGIT( buffer ) ) GOTO 5040
* *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	   READ ( buffer, *, ERR=5040 ) n
#else
	   READ ( buffer, '(I)', ERR=5040 ) n
#endif
	   chg_window = n .NE. wsid
	ELSE
	   chg_window = .FALSE.
* ... locate currently active window
	   DO 55 wn = 1, max_windows
	      IF ( wn_active( wn ) ) THEN
	         n = wn
	         GOTO 60
	      ENDIF
 55	   CONTINUE
	   GOTO 5060
	ENDIF

* valid number ?
* *kob* 1/00 - check to make sure user doesnt open window 9 - reserved
*              for animation purposes
 60	IF ( (n.LT.1 .OR. n.GT.(max_windows-1)) .AND.
     .          .NOT.animate  ) CALL ERRMSG
     .		( ferr_out_of_range, status, 'legal numbers are 1-8', *5000 )

* Determine the graphics engine to use.  This also sets the default
* anti-alias state for that engine.  If a new window and neither
* /quality nor /engine is given, the default engine is assigned.
        qp = qual_given(slash_quality)
        IF ( qp .GT. 0 ) THEN
* /QUALITY=... given; translate to an engine name to use,
* but do not force it to be used (in case -nodisplay was specified)
            CALL EQUAL_STRING(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                        buffer, status)
            IF ( status .NE. ferr_ok ) RETURN
            IF ( buffer .EQ. ' ' ) THEN
                CALL ERRMSG(ferr_syntax, status, '/QUALITY=?', *5000)
            ENDIF
            slen = TM_LENSTR1(buffer)
            IF ( MATCH4(buffer, slen, 'DRAFT', 5) ) THEN
                IF ( .NOT. FGD_ENGINE_MATCH(n, 'PipedImager') ) THEN
                    physical_chg = .TRUE.
                    CALL CANCEL_WINDOW_NUM(n)
                    CALL FGD_SET_ENGINE(n, 'PipedImager', .FALSE., status)
                    IF ( status .NE. ferr_ok ) RETURN
                ENDIF
            ELSE IF ( MATCH4(buffer, slen, 'HIGH', 4) ) THEN
                IF ( .NOT. FGD_ENGINE_MATCH(n, 'PipedViewerPQ') ) THEN
                    physical_chg = .TRUE.
                    CALL CANCEL_WINDOW_NUM(n)
                    CALL FGD_SET_ENGINE(n, 'PipedViewerPQ', .FALSE., status)
                    IF ( status .NE. ferr_ok ) RETURN
                ENDIF
            ELSE
                CALL ERRMSG(ferr_syntax, status,
     .                      'Unknown /QUALITY value of ' //
     .                      buffer(1:slen), *5000)
            ENDIF
            qp = qual_given(slash_engine)
            IF ( qp .GT. 0 ) THEN
* /ENGINE=... also given; ignore but warn
                warnmsg = '/QUALITY and /ENGINE both given ' //
     .                    'with SET WINDOW; /ENGINE ignored'
                CALL WARN(warnmsg)
            ENDIF
        ELSE
            qp = qual_given(slash_engine)
            IF ( qp .GT. 0 ) THEN
* /ENGINE=... given; force the use of this engine
                CALL EQUAL_STRING(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                         buffer, status)
                IF ( status .NE. ferr_ok ) RETURN
                slen = TM_LENSTR1(buffer)
                IF ( .NOT. FGD_ENGINE_MATCH(n, buffer(1:slen)) ) THEN
                    physical_chg = .TRUE.
                    CALL CANCEL_WINDOW_NUM(n)
                    CALL FGD_SET_ENGINE(n, buffer(1:slen), .TRUE., status)
                    IF ( status .NE. ferr_ok ) RETURN
                ENDIF
            ELSE IF ( .NOT. wn_open(n) ) THEN
* Neither /QUALITY=... nor /ENGINE=... given but a new window
* Give a blank engine name to reset to the default engine
                CALL FGD_SET_ENGINE(n, '', .TRUE., status)
                IF ( status .NE. ferr_ok ) RETURN
            ENDIF
        ENDIF

* create new window ?
	IF ( .NOT. wn_open(n) ) THEN
	   xwind_id = GKS_X_CONID( n )
	   CALL NEW_GKS_WINDOW( n, xwind_id, window_title(:tlen), status )   ! status always OK
	   physical_chg = .TRUE.
* ... set up GKS bundles for the new "workstation" (window)
	   CALL WS_LINE_BUNDLES( n, wstype )
           no_plot_yet = .TRUE.
	ENDIF

* clear the screen ? (which deletes all segments from window)
	IF ( qual_given( slash_clear ) .GT. 0 .AND. .NOT.animate) THEN
           CALL CLEAR_WINDOW( n )
	   vp_seg0( vp_num ) = curr_seg_name	! no segments left in viewport
	   vp_segn( vp_num ) = curr_seg_name
	ENDIF

* store appropriate segment range with each viewport
	IF ( chg_window ) THEN
* ... change windows --> segments will remain until previous window is cleared
	   vp_num = mvp_dflt
           PPL_viewports_active = .FALSE.    ! communicate to PPLUS (10/92)
	   DO 80 ivp = mvp_dflt, max_viewport
	      vp_seg0(ivp) = 0
	      vp_segn(ivp) = 0
 80	   CONTINUE
	ENDIF

* Decipher /SIZE /ASPECT /XPIXELS /YPIXELS /XINCHES /YINCHES
        CALL WINDOW_SIZE_FROM_OPTS(n, xinches, yinches,
     .              xpixels, ypixels, physical_chg, status)
        IF ( status .NE. ferr_ok ) RETURN

        IF ( physical_chg ) THEN
* tell ppl of the new default sizes for this window
           wn_xinches(n) = xinches
           wn_yinches(n) = yinches
           CALL SET_PPL_VALUES(n, .TRUE.)
        ELSEIF ( chg_window ) THEN
* tell ppl of the sizes used for this window
           CALL SET_PPL_VALUES(n, .FALSE.)
        ENDIF

* define the default normalization transformation for this window
        IF ( physical_chg .OR. chg_window ) THEN
           ASIZE = width * 1000
           BSIZE = height * 1000
           IF(ASIZE.GT.BSIZE)THEN
              FCTY=BSIZE/ASIZE
              FCTX=1.0
           ELSE
              FCTX=ASIZE/BSIZE
              FCTY=1.0
           ENDIF
           CALL FGD_GSWKWN(n, rzero, FCTX, rzero, FCTY)
           CALL FGD_GSVP(n, rzero, FCTX, rzero, FCTY)
           CALL FGD_GSWN(n, rzero, ASIZE, rzero, BSIZE)
        ENDIF

* deactivate previously active window
        DO 100 wn = 1, max_windows
           IF ( wn_active( wn ) ) THEN
              CALL FGD_GDAWK( wn )
	      wn_active( wn ) = .FALSE.
	   ENDIF
 100	CONTINUE

* activate window and select the default transform (same as the windowid)
        CALL FGD_GACWK(n)
        wsid = n
        wn_open  ( wsid ) = .TRUE.
        wn_active( wsid ) = .TRUE.

        IF ( physical_chg ) THEN

* ... clear previous plot and any viewports used within it
*     and set the size of the window.
*          Set ASIZE and BSIZE to invalid values so the SIZE routine
*          in SET_VIEWPORT will not return immediately, thinking there
*          was no size change from the ASIZE and BSIZE values set above.
           ASIZE = 0.0
           BSIZE = 0.0
           CALL SET_VIEWPORT( mvp_dflt )

*          Send the scaling factor to the viewer so the image is
*          the desired pixel size.  To make up for integer rounding,
*          compute the scaling factor using both height and width.
           xsc = xpixels / (xinches * windowdpix(n))
           ysc = ypixels / (yinches * windowdpiy(n))
*          negative value means set the scale to the absolute value
*          and then turn on auto-scaling
           sc = -1.0 * SQRT(xsc * ysc)
           CALL FGD_SEND_IMAGE_SCALE(n, sc)

*          Save xpixels and ypixels for specifying the default size
*          when saving raster graphics as well as values for
*          PPL$XPIXEL and PPL$YPIXEL
           wn_xpixels(n) = xpixels
           wn_ypixels(n) = ypixels

        ENDIF

* use anti-aliasing?
        qp = qual_given(slash_aalias)
        IF ( qp .GT. 0 ) THEN
* turn anti-aliasing on for this window
            CALL FGD_SET_ANTIALIAS(n, 1)
            qp = qual_given(slash_noaalias)
            IF ( qp .GT. 0 ) THEN
* also specified turning anti-aliasing off; ignore but warn
                warnmsg = 'anti-aliasing and no anti-aliasing ' //
     .            'both specified with SET WINDOW; using anti-aliasing'
                CALL WARN(warnmsg)
            ENDIF
        ELSE
            qp = qual_given(slash_noaalias)
            IF ( qp .GT. 0 ) THEN
* turn anti-aliasing off for this window
                CALL FGD_SET_ANTIALIAS(n, 0)
            ENDIF
        ENDIF

* set line thickness scaling factor?
        qp = qual_given(slash_thicken)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     scl, status)
            IF ( status .NE. ferr_ok ) RETURN
            IF ( scl .LE. 0.0 ) GOTO 5070
            dfltthick = scl
            CALL FGD_SET_THICK_FACTOR(n, dfltthick)
        ENDIF

* set text size scaling factor?
        qp = qual_given(slash_textprom)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     scl, status)
            IF ( status .NE. ferr_ok ) RETURN
            IF ( scl .LE. 0.0 ) GOTO 5090
*           Note: scale text by this factor as well as by pplscale
            CALL SET_TEXT_SIZES(n, scl)
        ENDIF

* set polygon/rectangle outline width?
        qp = qual_given(slash_outline)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     scl, status)
            IF ( status .NE. ferr_ok ) RETURN
            IF ( scl .LT. 0.0 ) GOTO 5100
            owd = scl
            CALL FGD_SET_OUTLINE_WIDTH(n, owd)
        ENDIF

* set scaling factor used in the viewer?
        qp = qual_given(slash_scale)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     scl, status)
            IF ( status .NE. ferr_ok ) RETURN
            sc = scl
            CALL FGD_SEND_IMAGE_SCALE(n, sc)
        ENDIF

* background color will always use color number zero
        bkgcolor = 0
* background color specified?
        qp = qual_given(slash_color)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_COLOR(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                       bkgcolor, dfltthick, status)
            IF ( status .NE. ferr_ok ) RETURN
*           Check if opacity is given before resetting the background
        ENDIF

* opacity specified? (must be after /COLOR to reset alpha channel value)
        qp = qual_given(slash_opacity)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     scl, status)
            IF ( status .NE. ferr_ok ) RETURN
            IF ( (scl .LT. 0.0) .OR. (scl .GT. 100.0) ) GOTO 5080
            opaqfrac = scl / 100.0
*           create a temporary color with the new opacity for the active window
            CALL DUP_COLOR_SET_OPACITY(n, bkgcolor, opaqfrac)
        ENDIF


* wmark specified? pass along values specified in /WMOPAC /WMSCALE /WMXLOC /WMYLOC
        qp = qual_given(slash_wmark)
        IF ( qp .GT. 0 ) THEN
            CALL EQUAL_STR_LC(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                        buffer, status)
            IF ( status .NE. ferr_ok ) RETURN
            s1 = TM_LENSTR(buffer)

            qp_op = qual_given(slash_wmark_opac)
            IF ( qp_op .GT. 0 ) THEN
                CALL EQUAL_VAL(cmnd_buff(qual_start(qp_op):qual_end(qp_op)),
     .                         opac, status)
                IF ( status .NE. ferr_ok ) RETURN
                IF ( (opac .LT. 0.0) .OR. (opac .GT. 100.0) ) GOTO 5080
                opac = opac / 100.0
            ELSE
               opac = 0.5
            ENDIF

            qp_scl = qual_given(slash_wmark_scale)
            IF ( qp_scl .GT. 0 ) THEN
                CALL EQUAL_VAL(cmnd_buff(qual_start(qp_scl):qual_end(qp_scl)),
     .                         scale, status)
                IF ( status .NE. ferr_ok ) RETURN
            ELSE
                scale = 1.0
            ENDIF

            qp_xl = qual_given(slash_wmark_xloc)
            IF ( qp_xl .GT. 0 ) THEN
                CALL EQUAL_VAL(cmnd_buff(qual_start(qp_xl):qual_end(qp_xl)),
     .                         wmkx, status)
                IF ( status .NE. ferr_ok ) RETURN
            ELSE
                wmkx = 0.0
            ENDIF

            qp_yl = qual_given(slash_wmark_yloc)
            IF ( qp_yl .GT. 0 ) THEN
                CALL EQUAL_VAL(cmnd_buff(qual_start(qp_yl):qual_end(qp_yl)),
     .                         wmky, status)
                IF ( status .NE. ferr_ok ) RETURN
            ELSE
                wmky = 0.0
            ENDIF

            xloc = wmkx
            yloc = wmky
            scalefrac = scale
            opacity = opac

            CALL FGD_SET_WMARK(n, buffer, s1, xloc, yloc, scalefrac, opacity)
        ENDIF

        IF ( bkgcolor > 0 ) THEN
*           Get the RGBA fractions for the new background
            CALL FGD_GQCR(n, bkgcolor, ier,
     .                    redfrac, grnfrac, bluefrac, opaqfrac)
*           Modify color zero and redraw the background
            CALL FGD_GSCR(n, 0, redfrac, grnfrac, bluefrac, opaqfrac)
            CALL FGD_REDRAW_BACKGROUND(n)
        ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exit(s)
 5000	RETURN
! 5010	CALL ERRMSG( ferr_erreq, status, buffer, *5000 )
 5020	CALL ERRMSG( ferr_syntax,status, buffer, *5000 )
 5040	CALL ERRMSG( ferr_unknown_arg, status, buffer, *5000 )
 5050	CALL ERRMSG( ferr_out_of_range, status, buffer,*5000)
 5060	CALL ERRMSG( ferr_invalid_command,status,
     .          'no open window',*5000 )
 5070   CALL ERRMSG( ferr_out_of_range, status,
     .          'Invalid thicken specification; '//
     .          'a positive number must be given', *5000)
 5080   CALL ERRMSG( ferr_out_of_range, status,
     .          'Invalid opacity specification; '//
     .          'a value from 0 to 100 must be given', *5000)
 5090   CALL ERRMSG( ferr_out_of_range, status,
     .          'Invalid text prominence specification; '//
     .          'a positive number must be given', *5000)
 5100   CALL ERRMSG( ferr_out_of_range, status,
     .          'Invalid outline width specification; '//
     .          'a non-negative number must be given', *5000)
	END
