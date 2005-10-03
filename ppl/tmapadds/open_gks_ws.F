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
*	Routine to open GKS if not open, the workstation.
*	Taken out of SIZE2 and modified for metafile.
*	J Davison JISAO/PMEL/TMAP
*	3.15.89
*	12.20.89 *sh* - added GKS_X_CONID code for FERRET
*       03.25.91 *jd* - mod to separate ws & metafile opening into two
*       subroutines rather than 1 with 2 entry points
*	*jd* 3.10.92 Mod to handle ATC GKS
*	*jd* 8.6.93 Mod to use XGKS
*       *sh* 3.4.97 (& 7/97) Mod to support metafile-only (no X) output
*       through a cuckoo's egg implantation of meta_wsid in place of wsid
*            8-4-97 -- substituted this edited version which didn't get
*            into CVS. This version calls open_gks_ws as long as not in
*            batch mode.
*  v5.4 *acm* 8/2001  add polymarker call for PLOT/SYMBOL=DOT 
*             here  set polymarker area attributes to bundled.
*      *acm* 10/01  remove VMS includes
*  v5.90 *acm* 9/05 append window_title to window name. If no title
*                   has been defined, set it to the session date:time.
*                   Save window_title in xplot_setup.cmn for use in subsequent 
*                   windows. Also define a symbol WIN_TITLE


	integer 	TM_LENSTR1, nlen, asf(13),error
	LOGICAL         ITS_BATCH_GRAPHICS, batch

        character*88    window_name

	include 'pltcom_dat.decl'! with declarations    
	include 'PLTCOM.DAT'! with declarations    
	include 'gkscm1_inc.decl'! with declarations    
	include 'GKSCM1.INC'! with declarations    
	include 'gkscm2.cmn'! with declarations    
	include 'ppl_in_ferret.cmn'
        include 'ws_types.cmn'
        include 'shade_vars.cmn'
        include 'xtoday.cmn'            ! with today_date and today_time
        include 'xplot_setup.cmn'

#ifdef atc_gks
#   ifdef unix
	include        'gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
	character	GKS_X_CONID*16, xwind_id*16
	external	gks_x_conid
#   ifdef unix
  	include        'gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif
#ifdef atc_gks	
        character*80     conid_options
        character*80     conid_opt_def/' -nc 64 -ox 2 -u 2'/
#endif

#ifdef sun
        BYTE      NULL
        BYTE      cstring(64) ! c-type Hollerith string buffer
#else
        INTEGER*1 NULL
        INTEGER*1 cstring(64) ! c-type Hollerith string buffer
#endif
	PARAMETER ( NULL =  0)


******************************************************************************** initialize
	batch = ITS_BATCH_GRAPHICS()	! 7/97
	IF (batch) THEN
	   CALL GOPKS(1,-1)  
!	   xwind_id = GKS_X_CONID( wsid )	! obsolete?
	   meta_wsid = wsid  ! cuckoo's egg -- output ws will be metafile
* ... metafile will be opened and activated later
	   GOTO 500
	ENDIF

*	Opens GKS and workstation, and activates it

C   
C       OPEN AND INITIALIZE GKS     
C   
#ifdef atc_gks
	call gopks (6,0)
        conid  = 1

	if (ppl_in_ferret) then
	   conid_opt_def = '-l FERRET 1' // conid_opt_def
	else
	   conid_opt_def = '-l PPLP' // conid_opt_def
	endif

#   ifdef unix
	call getenv ('atcGKSconid',conid_options)
#   else
*	GET LOGICAL WITH TRNLNM
#   endif

	if (wstype/100 .eq. ws_xwindow/100 .and. conid_options .eq. ' ') 
     .		conid_options = conid_opt_def

******************TEST X OPEN*****************
*
*        call openxwin (640,512,ixwin,ierr)
*        if (ierr .ne. 0) stop
*
*        write (unit=conid_options,15) ixwin
* 15     format ('-w',i8,'-sw -dp -cc')
*
****************END TEST 3.20.92**************

	if (conid_options .ne. ' ') then
           call guesc050 (conid,conid_options)
           call guesc005 (conid)
	endif
	call gopwk (wsid,conid,wstype)
#else
#ifndef xgks
	CALL GOPKS(ERRFIL)  
#else
	CALL GOPKS(1,-1)  
#endif
	if (wstype .eq. ws_tek4014 .or. wstype .eq. ws_tek4107) then
*	   open (ws_lun,file=ws_file,status='old')
*	   call gopwk (wsid,ws_lun,wstype)
           CALL GOPWK(WSID, 0, 0)
	elseif ( PPL_in_FERRET .and. (wstype. eq. ws_xwindow
     .				 .or. wstype .eq. 0)) THEN
* ... remote node connection option for FERRET
	   xwind_id = GKS_X_CONID( wsid )
#   ifdef unix
#ifndef xgks
	CALL GOPWK(WSID, 0, 0)
#else
	if (ppl_in_ferret) then

* If the window title has not yet been initialized, do so using the
* date and time of the Ferret session.

           clen = 64
	   CALL TM_FTOC_STRNG( window_title, cstring, clen )
           IF (cstring(1) .EQ. NULL) 
     .        window_title = today_date//':'//today_time
           nlen = TM_LENSTR1(window_title)

           IF (nlen .GT. 1 .OR. window_title .NE. '_')
     .       window_name = 'FERRET_1'//'_'//window_title
           ier = 0
           CALL SETSYM ("WIN_TITLE "//window_title(:nlen), nlen+10, ier, status)

	   call gesspn (window_name)
	else
	   call gesspn ('PPLP')
	endif
  	CALL GOPWK(WSID,6,4)       
#endif
#   else
	   CALL GKS$OPEN_WS( wsid, xwind_id, 0 )
#   endif
	else
#ifndef xgks
	   CALL GOPWK(WSID,CONID,WSTYPE)       
#else
	if (ppl_in_ferret) then
	   call gesspn ('FERRET_1',8)
	else
	   call gesspn ('PPLP',4)
	endif
	   CALL GOPWK(WSID,6,4)       
#endif
	endif
#endif
	CALL GACWK(WSID)    
    
*       SET DEFERRAL STATE TO GENERATE OUTPUT AS SOON AS POSSIBLE   
	call gsds (wsid,gasap,gsuppd) 

 500	CONTINUE	! 7/97 -- from here on BATCH mode, too

	GKSOPN=.TRUE.       
	gks_open = .true.  ! another open flag needed, PLTCOM.DAT has problems
	LPEN=.NOT.PEN       

*	SET POLYLINE AND FILL AREA ATTRIBUTES BUNDLED
	call gqasf (error,asf)				  
	if (error .ne. 0) goto 1000			     
								
	asf(1)  = gbundl					 
	asf(2)  = gbundl					 
	asf(3)  = gbundl		

*	SET POLYMARKER AREA ATTRIBUTES BUNDLED *acm* 8/2001
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
           call gsfais (gsolid)	
        endif
	call gsasf (asf)
	
*	SET UP DEFAULT LINE BUNDLES FOR WS
	IF ( .NOT.batch ) call setup_line_bundles (wsid,wstype) 
*ATC two lines following -- commented out for now
*	call gscr (wsid,0,1.0,1.0,1.0)
*	call gscr (wsid,1,0.0,0.0,0.0)

	return

*	FATAL ERROR -- NEED GKS ERROR HANDLING IN PPL...
1000	return

	end
