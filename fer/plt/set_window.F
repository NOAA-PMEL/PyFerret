      SUBROUTINE SET_WINDOW( status )

* create a new output window and de-activate (but don't remove) the last

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
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
*	is timing-related (plot must be complex, doesn't happen in totalview)
*	Changes here to clear window **before** other actions.
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/errmsg.parm'
        include 'tmap_pplv11inc/ppl_in_ferret.cmn' ! with PPL_viewports_active
	include	'tmap_pplv11inc/pltcom_dat.decl' 
	include	'pplv11inc/PLTCOM.DAT'         	  ! with asize,bsize
        include 'tmap_pplv11inc/plt_inc.decl'     ! with width/height
        include 'pplv11inc/PLT.INC'
	include	'tmap_pplv11inc/gkscm1_inc.decl'	! with gks common
	include	'pplv11inc/GKSCM1.INC'	                ! with gks common
	include 'tmap_pplv11inc/axis_inc.decl'		! axis lengths
	include 'pplv11inc/AXIS.INC'		        ! axis lengths
!	include 'sys_library/gksdefs.for'		! escape code
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
        INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN' ! with PPL_viewports_active
	INCLUDE	'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'
	INCLUDE	'PPLV11INC:PLTCOM.DAT/LIST'	  ! with asize,bsize
        INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'     ! with width/height
        INCLUDE 'PPLV11INC:PLT.INC'
	INCLUDE	'TMAP_PPLV11INC:GKSCM1_INC.DECL/LIST'	! with GKS common
	INCLUDE	'PPLV11INC:GKSCM1.INC/LIST'	        ! with GKS common
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'		! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'		        ! axis lengths
!	INCLUDE 'SYS_LIBRARY:GKSDEFS.FOR'		! escape code
#endif
	
* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	LOGICAL TM_DIGIT, physical_chg, relocate, chg_window,
     .		axis_ref
	INTEGER TM_LENSTR1, ier, id, itype, imeter, irx, iry, wn, n,
     .          qp, ivp, colon_pos
	REAL	dmaxx, dmaxy, fcty, fctx, xln, yln, xmn, xmx, ymn, ymx,
     .		bigness, scale, xstretch, ystretch, aspect,
     .		ref_len
	CHARACTER	UPPER_CASE*1, GKS_X_CONID*16,
     .			arg1*1, buffer*24, xwind_id*16, short_buff*24
* ... for GKS escape call:
	INTEGER esc_data(4), dummy
	DATA esc_data/1,0,0,-9/				!kob 1/97
* local parameter declarations:
	INTEGER		slash_size,
     .			slash_new,
     .			slash_loc,
     .			slash_aspect,
     .			slash_clear,
     .			slash_text
        LOGICAL         physical
	PARAMETER     ( slash_size   = 0+1,
     .			slash_new    = 1+1,
     .			slash_loc    = 2+1,
     .			slash_aspect = 3+1,
     .			slash_clear  = 4+1,
     .			slash_text   = 5+1,
     .                  physical     = .TRUE. )

* statement function - value between 0 and 1
	REAL x
	LOGICAL OUTSIDE_0_1
	OUTSIDE_0_1( x ) = x.LT.0.0 .OR. x.GT.1.0

* main window opened yet ?
	IF ( .NOT.pplus_started ) THEN
	   CALL START_PPLUS
	ENDIF

* are we in a valid GKS window environment ?
	IF ( .NOT. gksopn	) RETURN

* get number
	IF ( num_args .EQ. 0 .AND. qual_given(slash_new) .GT. 0 ) THEN
* ... find next free number
	   chg_window = .TRUE.
	   DO 50 n = 1, max_windows
	      IF ( .NOT.wn_open( n ) ) GOTO 60
 50	   CONTINUE
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
 60	IF ( n.LT.1 .OR. n.GT.max_windows ) CALL ERRMSG
     .		( ferr_out_of_range, status, 'legal numbers are 1-8', *5000 )

* clear the screen ? (which deletes all segments from window)
* (moved from near end of routine 
	IF ( qual_given( slash_clear ) .GT. 0 ) THEN
	   CALL CLEAR_WINDOW( n )
! segments already deleted inside CLEAR_WINDOW 9/95
!	   DO 1010 ivp = mvp_dflt, max_viewport
!	      vp_seg0(ivp) = 0
!	      vp_segn(ivp) = 0
! 1010	   CONTINUE
	   vp_seg0( vp_num ) = curr_seg_name	! no segments left in viewport
	   vp_segn( vp_num ) = curr_seg_name
	ENDIF
	   
* /SIZE=q
	qp = qual_given( slash_size )
	physical_chg = qp .GT. 0
	IF ( physical_chg ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   scale, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( scale .LE. 0.0 ) GOTO 5050
	   wn_scale(n) = SQRT( scale )
	ENDIF

* /ASPECT=ratio[:"AXIS" or "EDGE"(default)]
	qp = qual_given( slash_aspect )
	IF ( qp .GT. 0 ) THEN
* ... decode the ratio and possible ":SIZE" or ":AXES" qualifier
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buffer, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buffer .EQ. ' ' ) THEN
	      buffer = 'aspect ratio ?'
	      GOTO 5050
	   ENDIF
	   colon_pos = INDEX( buffer,':' )
	   IF ( colon_pos .GT. 0 ) THEN
	      short_buff = buffer(:colon_pos-1)			! for AIX
	      READ ( short_buff,* ,ERR=5020 ) aspect
	      arg1 = UPPER_CASE( buffer(colon_pos+1:colon_pos+1) )
	      axis_ref = arg1 .EQ. 'A'
	      IF ( .NOT.(axis_ref .OR. arg1 .EQ. 'E') ) GOTO 5040
	   ELSE
	      READ ( buffer, * ,ERR=5020 ) aspect
	      axis_ref = .FALSE.    ! default to "edge"  (10/9/91)
	   ENDIF
	   IF ( aspect.LE.0.0 ) GOTO 5050
* ... set up the plot preserving plot area relative to default 6x8 plot
* ... allowing an additional 2.2 and 2.8 in x and y resp. for labelling
	   ystretch = SQRT( aspect )
	   xstretch = 1.0 / ystretch
	   IF ( axis_ref ) THEN
	      ref_len = SQRT( 8.0 * 6.0 )
	      CALL SET_PLOT_SIZE( xstretch*ref_len + 2.2,
     .                            ystretch*ref_len + 2.8 )
	   ELSE
	      ref_len = SQRT( 10.2 * 8.8 )
	      CALL SET_PLOT_SIZE( xstretch*ref_len, ystretch*ref_len )
	   ENDIF
* ... clear previous plot and any viewports used within it
	   IF ( .NOT.chg_window ) CALL SET_VIEWPORT( mvp_dflt )
	ENDIF

! note: /TEXT_PROMINENCE will not perform correctly within viewports as it
! reshapes the windows
!* /TEXT_PROMINENCE=factor
!* ... works by blowing up plot size in Denbo inches while holding labels and
!*     label positions relative to plot fixed
!	qp = qual_given( slash_text )
!	IF ( qp .GT. 0 ) THEN
!	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
!     .			   text_prom, status )
!	   IF ( status .NE. ferr_ok ) RETURN
!	   IF ( aspect.LE.0.0 ) GOTO 5050
!	   text_prom = SQRT( text_prom )
!	   xstretch = xstretch / text_prom
!	   ystretch = ystretch / text_prom
!	   wn_scale(n) = wn_scale(n) * text_prom
!	ENDIF
!
!* communicate /ASPECT and /TEXT_PROMINENCE to PPLUIS
!	CALL SET_PLOT_SIZE( xstretch, ystretch )

* lower left location specified ?
	qp = qual_given( slash_loc )
	relocate = qp .GT. 0
	IF ( relocate ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buffer, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.TM_DIGIT( buffer ) )  GOTO 5030	! need a [0,1] value
	   IF ( INDEX(buffer,',') .EQ. 0 ) GOTO 5030
	   READ ( buffer, *, ERR=5030 ) xmn, ymn	! '*' not removed 10/95
	   IF ( OUTSIDE_0_1(xmn) .OR. OUTSIDE_0_1(ymn) ) GOTO 5030
	   physical_chg = .TRUE.
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
           CALL SET_PLOT_SIZE( width, height )   ! from PPL COMMON PLT.INC
           CALL SET_AX_SIZES( 1.0, 1.0 )
	ENDIF

* create new window ?
	IF ( .NOT. wn_open(n) ) THEN
	   xwind_id = GKS_X_CONID( n )
	   CALL NEW_GKS_WINDOW( n, xwind_id, status )   ! status always OK
	   physical_chg = .TRUE.
* ... set up GKS bundles for the new "workstation" (window)
	   CALL SETUP_LINE_BUNDLES( n, wstype )
	ELSE
	   CONTINUE
! ... doesn't work:
! "%GKS-E-ERROR_180, Specified escape function is not supported in routine GESC"
!	   esc_data(4) = %LOC(n)
!	   CALL GKS_ESCAPE( GKS_K_ESC_POP_WORKSTATION, esc_data, 16,
!     .			    dummy, dummy, dummy )
	ENDIF

* deactivate previously active window
	DO 100 wn = 1, max_windows
	   IF ( wn_active( wn ) ) THEN
	      CALL GDAWK( wn )
	      wn_active( wn ) = .FALSE.
	   ENDIF
 100	CONTINUE

* activate window
	 CALL GACWK(n)

*set up sizing
* (capitalized code from Denbo's SUBROUTINE COLOR )
	IF(ASIZE.GT.BSIZE)THEN
	   FCTY=BSIZE/ASIZE
	   FCTX=1.0
	ELSE
	   FCTX=ASIZE/BSIZE
	   FCTY=1.0
	ENDIF
* ... define normalization transformation n (actually selected down below)
	IF ( chg_window ) THEN
	   CALL GSWN(n,0.0,ASIZE,0.0,BSIZE)
	   CALL GSVP(n, 0.0, FCTX, 0.0, FCTY )
	ENDIF

	IF ( .NOT.physical_chg ) GOTO 1000
c
c	set up the size of the display window
c
	    CALL GQWKC(n,IER,ID,ITYPE)
	    CALL QUERY_DISPLAY_SIZE(ITYPE,IER,IMETER,
     .                              DMAXX,DMAXY,IRX,IRY)

* special location specified on [0,1] scale ?
	    IF ( relocate ) THEN
	      xlowlef(n) = xmn * dmaxx
	      ylowlef(n) = ymn * dmaxy
	    ENDIF

	    IF(WKSAUT)THEN
		XLN=ASIZE*2.54E-5	! Denbo milli-inches to meters
		YLN=BSIZE*2.54E-5

* re-scale as requested
	        xln = wn_scale(n) * xln
	        yln = wn_scale(n) * yln

* ... too big to fit ?
	        bigness = MIN( dmaxx/xln, dmaxy/yln )
	        IF ( bigness .LT. 1.0 ) THEN
	           xln = xln * bigness
	           yln = yln * bigness
	        ENDIF

* ... put lower left of each window at a different place if possible
	        wkxmn = MIN( xlowlef(n), dmaxx-xln )
	        wkymn = MIN( ylowlef(n), dmaxy-yln )
	        	        
		WKXMX=WKXMN+XLN
		WKYMX=WKYMN+YLN
C
C	SET THE NDC WINDOW
C
		CALL GSWKWN(n,0.0,FCTX,0.0,FCTY)
C
C	SET DEVICE (WORKSTATION) VIEWPORT
C
		CALL SET_WINDOW_SIZE(n,WKXMN,WKXMX,WKYMN,WKYMX)
	    ELSE
C
C	IF NOT AUTO WKXMN AND WKXMX ETC ARE FRACTIONS OF DMAXX
C	AND DMAXY
C
		CALL GSWKWN(n,0.0,FCTX,0.0,FCTY)
		XMN=DMAXX*WKXMN
		XMX=DMAXX*WKXMX
		YMN=DMAXY*WKYMN
		YMX=DMAXY*WKYMX
		CALL SET_WINDOW_SIZE(n,XMN,XMX,YMN,YMX)
	    ENDIF

* force immediate creation of the window on the screen
	CALL GUWK( n, 1 )

* turn on normalization transform (same as window number and wsid)
 1000	CALL GSELNT(n)

* successful completion
	wsid = n
	wn_open  ( wsid ) = .TRUE.
	wn_active( wsid ) = .TRUE.
	status = ferr_ok
	RETURN

* error exit(s)
 5000	RETURN
! 5010	CALL ERRMSG( ferr_erreq, status, buffer, *5000 )
 5020	CALL ERRMSG( ferr_syntax,status, buffer, *5000 )
 5030	CALL ERRMSG( ferr_out_of_range,status,
     .		'need x and y location between 0 and 1:"'
     .		//buffer(:TM_LENSTR1(buffer))//'"',*5000)
 5040	CALL ERRMSG( ferr_unknown_arg, status, buffer, *5000 )
 5050	CALL ERRMSG( ferr_out_of_range, status, buffer,*5000)
 5060	CALL ERRMSG( ferr_invalid_command,status,
     .          'no open window',*5000 )
	END

