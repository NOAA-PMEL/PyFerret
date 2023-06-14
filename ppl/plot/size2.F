      SUBROUTINE SIZE(width,height)

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
* revised for FERRET
* from PPLUS :

*       Directory PMEL:[PLOT5.LIBRARY.PLOT]

*       SIZE2.FOR;14              9   5-APR-1987 12:26

* search for "FERRET" to find all revisions
* *sh* 6/30/88 - rev 1.0 for FERRET 1.00+
*       - changed GKS size and location calculations to conform
*       - to SET WINDOW in FERRET
*       - note: the changes here are to correct a bug that SET WIND/SIZE=n 1
*               is ignored if issued before the first GKS graphical output
* *sh* 8/16/88 - rev 1.1 for FERRET 1.10
*       - added immediate update of display after re-sizing
* *sh* 8/26/88 - rev 1.1 for FERRET 1.10+
*       - added code for SET WINDOW/VIEWPORT in FERRET
* *jd* 9.7.88  - rev 2.0
*       - Modified to properly handle those workstations (like TEK) that
*               do not use meters in their device coordinates and those
*               requiring ASAP deferral state
* *sh* 9/13/88 - rev 2.1 for FERRET 1.10+
*       - allow clipping flagged as OFF via vp_*clip.EQ.unspecified_val4
* *sh* 9/29/88 - rev 2.2 for FERRET 1.10+
*       - allow separate normalization transformation for each wsid (#=wsid)
*         instead of all windows sharing #1
* *jd* 11.28.88
*       - Modified for compatibility to version 1.1 of PPL
* *sh* 9/29/88
*       - further PPLUS V11 compatibility mods - eliminated FERRET common
* *jd* 03.15.89
*       - Mod to improve GKS control -- removed GKS open and transferred to
*         GKS_INIT_WS
* *jd* 10.26.89
*       - Mod to write workstation transform information to metafile
* *jd* 03.22.90
*       - Mod to accomodate having no open ws, but metafiles open, in Ferret
* *jd* 03.25.92 
*       - Mod to accommodate ATC GKS -- replaced GQDSP & GSWKVP
* *jd* 10.26.93
*       - Mod to eliminate private item (wsxform) for xgks
* *sh* 4/9/97
*       - check to make sure only GKSM metafiles (not CGM) get
*         workstation sizing information
* *am* 1/07 V6.02 true batch mode for metafiles; flag meta_batch
*         distinguises this case from other batch modes.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus

*     PLOTTER DEPENDENT ROUTINE
*     CONTAINS PHYSICAL CHARACTERISTICS OF ALL LEGAL DEVICES.
*
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
C     CALL SIZE(WIDTH,HEIGHT)
C
C     THIS SUBROUTINE DEFINES THE DIMENSIONS OF THE TOTAL PLOT-
C     TING AREA.  SUBSEQUENT TO A CALL TO SIZE, PLOTTING IS AL-
C     LOWED ONLY WITHIN THESE LIMITS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        WIDTH  - TOTAL WIDTH OF THE PLOTTING AREA IN INCHES (OR
C                 MM).  WIDTH IS RESTRICTED TO A MAXIMUM OF 200
C                 INCHES ON THE ZETA PLOTTER AND 200 INCHES ON THE
C                 BENSON PLOTTER.
C        HEIGHT - TOTAL HEIGHT OF THE PLOTTING AREA IN INCHES
C                 (OR MM).  HEIGHT IS RESTRICTED TO A MAXIMUM OF
C                 34 INCHES ON THE ZETA PLOTTER AND 38 INCHES ON
C                 THE BENSON PLOTTER.
C
C     SIZE MAKES THE TEKTERMINAL AND PRINTRONIX PLOTTER APPEAR AS IF
C     THEY WERE WIDTH INCHES (OR MM) WIDE AND HEIGHT INCHES (OR
C     MM) HIGH.
C
C
* V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0

        include 'pltcom_dat.decl'       ! with declarations
        include 'PLTCOM.DAT'            ! with declarations
        include 'gkscm1_inc.decl'       ! with declarations
        include 'GKSCM1.INC'            ! with declarations
        include 'gkscm2.cmn'            ! metafile info
        include 'gkspar.inc'

* special COMMON to let this routine know if FERRET is in control
        include 'ppl_in_ferret.cmn'

        DATA ppl_in_ferret / .FALSE. /

        REAL width, height

        LOGICAL TM_FPEQ_SNGL
        LOGICAL clip
        INTEGER ier, imeter, irx, iry
        REAL    dmaxx, dmaxy, fcty, fctx, xln, yln, xmn, xmx, ymn, ymx,
     .          t, scale, tmxclip, tmyclip, tmxorg, tmyorg, tmsize,
     .          tmscale, tmxll, tmyll
*       REAL    bigness

        LOGICAL shrnkx,shrnky,first
        SAVE fctx,fcty !Needed for Ultrix
C
        IF ( (width .LE. 0.0) .OR. (height .LE. 0.0) ) RETURN

C
C     INSURE THAT THE PARAMETERS ARE POSITIVE AND CONVERT THEM
C     TO INTERNAL INCREMENTS.
C
        anew = factor * ABS(width)
        bnew = factor * ABS(height)

	IF ( TM_FPEQ_SNGL(ANEW,ASIZE) .AND. TM_FPEQ_SNGL(BNEW,BSIZE) ) THEN
	   RETURN
	end if
	ASIZE=ANEW
	BSIZE=BNEW
C
        IF (rotat) THEN
           t=asize
           asize=bsize
           bsize=t
        ENDIF
C
C     COMPUTE THE TEKTERMINAL CONVERSION FACTORS.
C
        IF (ttype .GT. 0) THEN
           IF (asize / twidth .GT. bsize / thight) THEN
              pltota = tscla / asize
              pltotb = tsclb / asize * twidth / thight
           ELSE
              pltota = tscla / bsize * thight / twidth
              pltotb = tsclb / bsize
           ENDIF
        ELSE
           pltota=tscla/(twidth*factor)
           pltotb=tsclb/(thight*factor)
        ENDIF
C
C     SET THE MINIMUM AND MAXIMUM ALLOWABLE VALUES OF A AND B.
C
        amin = 0.0
        bmin = 0.0
        amax = asize
        bmax = bsize
        IF (ptype .GT. 2) THEN
           IF (asize .GT. bsize) THEN
              fcty=bsize/asize
              fctx=1.0
           ELSE
              fctx=asize/bsize
              fcty=1.0
           ENDIF

* FERRET insert 9/1/88 - modified 9/13/88 - and 1/19/89
* use normalization transforms to allow viewports within output window

* default window and viewport values (full window)
           tmxorg  = 0.0
           tmyorg  = 0.0
           tmxclip = 1.0
           tmyclip = 1.0
           tmsize  = 1.0
           tmscale = 1.0        ! full sized
           tmxll   = 0.2
           tmyll   = 0.2
           clip    = .FALSE.

* FERRET in control ?
           IF ( ppl_in_ferret )
     .        CALL GET_VP_FROM_FERRET(tmxorg, tmyorg, tmsize, tmscale,
     .                                tmxclip, tmyclip, tmxll, tmyll)

* The following calls to GSVP, GSWN, GSELNT refer to wsid but are not
* restricted in action to a particular WS.  All WS use the same normalization
* transform, the active one, set by SELNT.  In particular the metafile output
* WS will be affected by these calls. jd 3.15.89
*

* set viewport in NDC to limits requested
* 9/29 - "1" --> wsid *sh*

           CALL FGD_GSVP(MAX(wsid,1),
     .                   fctx*tmxorg, fctx*tmxclip, 
     .                   fcty*tmyorg, fcty*tmyclip)

* set world coordinate window to preserve 1:1 aspect
           scale = 1.0 / SQRT( tmsize )
           CALL FGD_GSWN(MAX(wsid,1),
     .                   0.0, (tmxclip-tmxorg)*scale*asize,
     .                   0.0, (tmyclip-tmyorg)*scale*bsize)

* end of 9/1/88 and 9/13/88 FERRET mods
c
c set up the size of the display window
c          Note: FGD_GQDSP uses the window id 
c          instead of window type for the first argument
c
           CALL FGD_GQDSP(wsid,ier,imeter,dmaxx,dmaxy,irx,iry)
C
           IF ( wksaut ) THEN
              xln=asize*2.54E-5
              yln=bsize*2.54E-5

* FERRET mod: all lines below beginning with "!" have been commented out for
* FERRET: (ie. calculations for position and size of window have been changed)
!               FIRST=.TRUE.
!200            WKXMN=DMAXX-.0127-XLN
!               WKYMN=.00635
!C
!               SHRNKX=.FALSE.
!               SHRNKY=.FALSE.
!               IF(WKXMN.LT.0)THEN
!                   WKXMN=0.0
!                   YLN=YLN*DMAXX/XLN
!                   XLN=DMAXX
!                   SHRNKX=.TRUE.
!               ENDIF
!               IF(WKYMN+YLN.GT.DMAXY)THEN
!                   WKYMN=0.0
!                   XLN=XLN*DMAXY/YLN
!                   YLN=DMAXY
!                   SHRNKY=.TRUE.
!               ENDIF
!               IF(FIRST.AND.SHRNKX.AND.SHRNKY)THEN
!                   FIRST=.FALSE.
!                   GOTO 200
!               ENDIF
* FERRET insert: ************************************************************
* re-scale as requested by SET WINDOW command 
* do not do this anymore
*              xln = tmscale * xln
*              yln = tmscale * yln
*
* ... too big to fit ?
*             bigness = MIN( dmaxx/xln, dmaxy/yln )
*             IF ( bigness .LT. 1.0 ) THEN
*                xln = xln * bigness
*                yln = yln * bigness
*             ENDIF

* let the system window manager deal with placing the window on the screen
              wkxmn = 0.0
              wkymn = 0.0

* end of FERRET insert: ******************************************************
C
              wkxmx=wkxmn+xln
              wkymx=wkymn+yln
C
C SET THE NDC WINDOW

*             Next 5 lines *jd* 3.25.92
*              IF (.NOT. ppl_in_ferret) THEN
*                 CALL FGD_GSWN(1,0.0,asize,0.0,bsize)
*                 CALL FGD_GSVP(1,0.0,fctx,0.0,fcty)
*                 CALL FGD_GSELNT(1)
*              ENDIF

C
C SET DEVICE (WORKSTATION) VIEWPORT
C
              IF ( wsid .GT. 0 ) THEN
                 CALL FGD_GSWKWN(wsid,0.0,fctx,0.0,fcty)
                 CALL FGD_GSWKVP(wsid,wkxmn,wkxmx,wkymn,wkymx)
              ENDIF
           ELSE
C
C IF NOT AUTO WKXMN AND WKXMX ETC ARE FRACTIONS OF DMAXX AND DMAXY
C
              XMN=DMAXX*WKXMN
              XMX=DMAXX*WKXMX
              YMN=DMAXY*WKYMN
              YMX=DMAXY*WKYMX
              IF ( wsid .GT. 0 ) THEN
                 CALL FGD_GSWKWN(wsid,0.0,fctx,0.0,fcty)
                 CALL FGD_GSWKVP(wsid,xmn,xmx,ymn,ymx)
              ENDIF
           ENDIF

* select normalization transformation and turn on clipping
* -- after the final size of the window has been set --

           CALL FGD_GSELNT(MAX(wsid,1))
           IF ( clip ) THEN
              CALL FGD_GSCLIP(gclip)
           ELSE
              CALL FGD_GSCLIP(gnclip)
           ENDIF

* FERRET mod: force immediate update of display
           IF ( wsid .GT. 0 )
     .             CALL FGD_GUWK(wsid, 1)
        ENDIF
C
C     CALCULATE THE OFF-SCREEN FLAG.
C
        losflg = 0
        IF (aold - 0.001 .GT. amax) losflg = 1
        IF (aold + 0.001 .LT. amin) losflg = 2
        IF (bold - 0.001 .GT. bmax) losflg = losflg + 4
        IF (bold + 0.001 .LT. bmin) losflg = losflg + 8

        RETURN
C
        END
