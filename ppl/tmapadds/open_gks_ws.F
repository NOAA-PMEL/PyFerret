        SUBROUTINE OPEN_GKS_WS

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
*       Routine to open GKS if not open, the workstation.
*       Taken out of SIZE2 and modified for metafile.
*       J Davison JISAO/PMEL/TMAP
*       3.15.89
*       12.20.89 *sh* - added GKS_X_CONID code for FERRET
*       03.25.91 *jd* - mod to separate ws & metafile opening into two
*       subroutines rather than 1 with 2 entry points
*       *jd* 3.10.92 Mod to handle ATC GKS
*       *jd* 8.6.93 Mod to use XGKS
*       *sh* 3.4.97 (& 7/97) Mod to support metafile-only (no X) output
*       through a cuckoo's egg implantation of meta_wsid in place of wsid
*            8-4-97 -- substituted this edited version which didn't get
*            into CVS. This version calls open_gks_ws as long as not in
*            batch mode.
*  v5.4 *acm* 8/2001  add polymarker call for PLOT/SYMBOL=DOT
*             here  set polymarker area attributes to bundled.
*      *acm* 10/01  remove VMS includes
*  V600 *acm* 9/05 append window_title to window name. If no title
*                   has been defined, set it to the session date:time.
*                   Save window_title in xplot_setup.cmn for use in subsequent
*                   windows. Also define a symbol WIN_TITLE
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0


        IMPLICIT NONE
        integer      TM_LENSTR, TM_LENSTR1, s1, nlen, asf(13), error,
     .               clen, ier, status

        character*88 window_name

        include 'pltcom_dat.decl'! with declarations
        include 'PLTCOM.DAT'! with declarations
        include 'gkscm1_inc.decl'! with declarations
        include 'GKSCM1.INC'! with declarations
        include 'gkscm2.cmn'! with declarations
        include 'ppl_in_ferret.cmn'
        include 'ws_types.cmn'
        include 'shade_vars.cmn'
        include 'xtoday.cmn'            ! with today_date and today_time
	include 'tmap_dims.parm'
        include 'xplot_setup.cmn'

        character       GKS_X_CONID*16, xwind_id*16
        external        gks_x_conid
        include        'gkspar.inc'

        INTEGER*1 NULL
        INTEGER*1 cstring(64) ! c-type Hollerith string buffer
        PARAMETER ( NULL = 0 )


********************************************************************************
C
C       OPEN AND INITIALIZE GKS
C
        CALL FGD_GOPKS(1)
        IF (wstype .EQ. ws_tek4014 .OR. wstype .EQ. ws_tek4107) THEN
           CALL FGD_GOPWK(WSID, 0, 0)
        ELSE IF ( ppl_in_ferret .AND. ((wstype. EQ. ws_xwindow)
     .                            .OR. (wstype .EQ. 0)) ) THEN
* ... remote node connection option for FERRET
           xwind_id = GKS_X_CONID( wsid )
* If the window title has not yet been initialized, do so using the
* date and time of the Ferret session.
           clen = 64
           CALL TM_FTOC_STRNG(window_title, cstring, clen)
           IF ( cstring(1) .EQ. NULL ) THEN
              s1 = 1
              IF (today_date(1:1) .EQ. ' ') s1 = 2
              nlen = TM_LENSTR1(today_date)
              window_title = today_date(s1:nlen)//'_'//today_time
           ENDIF
           nlen = TM_LENSTR(window_title)
           IF ( (nlen .GT. 1) .OR.
     .         ((nlen .EQ. 1) .AND. (window_title .NE. '_')) )
     .        window_name = 'FERRET_1'//'_'//window_title
           ier = 0
           CALL SETSYM ("WIN_TITLE "//window_title(:nlen),
     .                  nlen+10, ier, status)
           CALL FGD_GESSPN(window_name)
           CALL FGD_GOPWK(WSID,6,4)
        ELSE IF ( ppl_in_ferret ) THEN
           CALL FGD_GESSPN('FERRET_1')
           CALL FGD_GOPWK(WSID,6,4)
        ELSE
           CALL FGD_GESSPN('PPLP')
           CALL FGD_GOPWK(WSID,6,4)
        ENDIF
        CALL FGD_GACWK(WSID)
        IF ( save_on_exit ) THEN
*          Inform the graphics engine of the image filename
           CALL OPEN_METAFILE
        ENDIF

*       SET DEFERRAL STATE TO GENERATE OUTPUT AS SOON AS POSSIBLE
        call FGD_GSDS(wsid,gasap,gsuppd)

        GKSOPN=.TRUE.
        gks_open = .true.  ! another open flag needed, PLTCOM.DAT has problems
        LPEN=.NOT.PEN

*       SET POLYLINE AND FILL AREA ATTRIBUTES BUNDLED
        call FGD_GQASF(error,asf)
        if (error .ne. 0) goto 1000

        asf(1)  = gbundl
        asf(2)  = gbundl
        asf(3)  = gbundl

*       SET POLYMARKER AREA ATTRIBUTES BUNDLED *acm* 8/2001
*       (see ws_line_bundld.F and ps_line_bundle.F)

        asf(4)  = gbundl
        asf(5)  = gbundl
        asf(6)  = gbundl

        if (area_bundles) then
           asf(11) = gbundl
           asf(12) = gbundl
           asf(13) = gbundl
        else
           asf(11) = gindiv
           asf(12) = gindiv
           asf(13) = gindiv
           call FGD_GSFAIS(gsolid)
        endif
        call FGD_GSASF(asf)

*       SET UP DEFAULT LINE BUNDLES FOR WS
        call ws_line_bundles (wsid,wstype)

        return

*       FATAL ERROR -- NEED GKS ERROR HANDLING IN PPL...
1000    return

        end

