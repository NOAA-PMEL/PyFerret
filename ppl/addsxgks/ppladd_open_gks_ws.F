	SUBROUTINE OPEN_GKS_WS

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

	integer 	asf(13),error
	LOGICAL         ITS_BATCH_GRAPHICS, batch

#ifdef unix
	include 'tmap_pplv11inc/pltcom_dat.decl'! with declarations    
	include 'pplv11inc/PLTCOM.DAT'! with declarations    
	include 'tmap_pplv11inc/gkscm1_inc.decl'! with declarations    
	include 'pplv11inc/GKSCM1.INC'! with declarations    
	include 'tmap_pplv11inc/gkscm2.cmn'! with declarations    
	include 'tmap_pplv11inc/ppl_in_ferret.cmn'
        include 'tmap_pplv11inc/ws_types.cmn'
        include 'tmap_pplv11inc/shade_vars.cmn'
#else
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'! with declarations    
	INCLUDE 'PPLV11INC:PLTCOM.DAT'! with declarations    
	INCLUDE 'TMAP_PPLV11INC:GKSCM1_INC.DECL'! with declarations    
	INCLUDE 'PPLV11INC:GKSCM1.INC'! with declarations    
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'! WITH DECLARATIONS    
	INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
        INCLUDE 'tmap_pplv11inc:ws_types.inc'
	INCLUDE 'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
	character	GKS_X_CONID*16, xwind_id*16
	external	gks_x_conid
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif
#ifdef atc_gks	
        character*80     conid_options
        character*80     conid_opt_def/' -nc 64 -ox 2 -u 2'/
#endif


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
	   call gesspn ('FERRET_1',8)
	else
	   call gesspn ('PPLP',4)
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
