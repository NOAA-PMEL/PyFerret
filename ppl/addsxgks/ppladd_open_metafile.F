	SUBROUTINE OPEN_METAFILE

*	Routine to open GKS metafile
*	J Davison JISAO/PMEL/TMAP
*	3.15.89
*	12.20.89 *sh* - added GKS_X_CONID code for FERRET
*       03.25.91 *jd* - took out of INIT_GKS_WS to have two subroutines
*       rather than 1 with 2 entry points
*       04.22.91 *jd* - mods to support new metafile translator
*       05.06.81 *jd* - mod to maintain metafile version #'s on ultrix
*       01.06.94 *jd*   Mod to put line color rep info into metafile for xgks
*       7.26.97  *js*   Restored call to set line bundles. Forced to be same
*                          as XGKS bundles. Added subroutine query_xgks_wkstn
*                          to support workstation queries on metafiles. Fixed
*                          bug where setup_line_bundles always called with type
*                           ws_xwindow

#ifdef unix
	include 'tmap_pplv11inc/pltcom_dat.decl'! with declarations    
	include 'pplv11inc/PLTCOM.DAT'! with declarations    
	include 'tmap_pplv11inc/gkscm1_inc.decl'! with declarations    
	include 'pplv11inc/GKSCM1.INC'! with declarations    
	include 'tmap_pplv11inc/gkscm2.cmn'! with declarations    
	include 'tmap_pplv11inc/ppl_in_ferret.cmn'
	include 'tmap_pplv11inc/ws_types.cmn'
#else
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'! with declarations    
	INCLUDE 'PPLV11INC:PLTCOM.DAT'! with declarations    
	INCLUDE 'TMAP_PPLV11INC:GKSCM1_INC.DECL'! with declarations    
	INCLUDE 'PPLV11INC:GKSCM1.INC'! with declarations    
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'! WITH DECLARATIONS    
	INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
	INCLUDE 'TMAP_PPLV11INC:WS_TYPES.CMN'
#endif

#ifdef unix
        integer       park_stat
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

	integer 	ndx,type,error,len,tm_lenstr
	real		red,green,blue

	integer		errind,errnr,fctid
	integer         theWsType

C	XGKS DOESN'T DEFINE GMOUTP 
	integer		gmoutp
	parameter	(gmoutp=2)


***************************************************************************

*	Opens metafile but does not activate it.

*	OPEN METAFILE WS 
#ifdef unix
        call tm_park_last_version (meta_file,park_stat)
#   ifndef atc_gks
	open (unit=meta_lun,file=meta_file,status='new',err=100)
#   endif
#else
	open (unit=meta_lun,file=meta_file,status='new',err=100,
     .	disp='delete')
#endif
#ifdef atc_gks
*	OPEN OUTPUT GKSM METAFILE
*       TEST TO SEE IF FILE NAME IS VALID -- ATC DOESN'T EXPLICITLY OPEN FILE
	open (unit=meta_lun,file=meta_file,status='new',err=100)
        close (meta_lun,status='delete')

	call guesc050 (meta_lun,meta_file)
	theWsType = 1002
	call gopwk(meta_wsid,meta_lun,1002)
#else
	theWsType = gmoutp
#endif
	call gopwk(meta_wsid,meta_lun,theWsType)
*	call gsds (meta_wsid,gasap,gsuppd) !Better to be 'At Some TIme'
	call gsds (meta_wsid,gasti,gsuppd) 

*	IS THE GUWK CALL NEEDED FOR THE METAFILE TOO? DON'T KNOW...
*       WILL TRY TO LIVE W/O IT
*	call guwk (meta_wsid,gpostp?gperfo)

*	SET UP DEFAULT LINE BUNDLES FOR METAFILE WS
*    -- PLACE NO LINE BUNDLE INFO IN METAFILE (4.22.91)
*    -- Changed back to placing line bundle info in metafile (7.26.97)
	call setup_line_bundles (meta_wsid,theWsType) 

	meta_open = .true.
!	meta_actv = .false.

#ifdef xgks
	if (meta_actv) then
*		ALLOW NO WHITE LINES IN METAFILE (FOR NOW)
                call gqcr (wsid,1,type,error,red,green,blue)
                if (error .eq. 0) then
		   if (red   .eq. 1.0 .and. 
     .		       green .eq. 1.0 .and.
     .		       blue  .eq. 1.0) then
                      call gscr (meta_wsid,1,0.0,0.0,0.0)
		    else
                      call gscr (meta_wsid,1,red,green,blue)
		    endif
                else
                   call gscr (meta_wsid,1,0.0,0.0,0.0)
                endif

                do ndx = 2,6
                   call gqcr (wsid,ndx,type,error,red,green,blue)
                   if (error .eq. 0) then
                      call gscr (meta_wsid,ndx,red,green,blue)
                   else
                      call gscr (meta_wsid,ndx,0.0,0.0,0.0)
                   endif
                end do
	end if
#endif
                                                        
 20     return

*	ERR open metafile
100	len = tm_lenstr (meta_file)  
#ifdef F90_SYSTEM_ERROR_CALLS
	call linux_perror('**ERROR: '//meta_file(1:len))
#else
	call perror ( '**ERROR: '//meta_file(1:len))
#endif
	return	
	
	end

c
c       Support queries on metafile workstations
c
	subroutine query_xgks_wkstn(zwstype,error,max_pline,max_pmark,
	1    max_text,max_fill_area,max_pattern,max_color)
	implicit none
#ifdef unix
	include 'tmap_pplv11inc/gkscm2.cmn'! with declarations    
	include 'tmap_pplv11inc/ws_types.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'! WITH DECLARATIONS    
	INCLUDE 'TMAP_PPLV11INC:WS_TYPES.CMN'
#endif
	integer zwstype,error,max_pline,max_pmark
	integer max_text, max_fill_area, max_pattern,max_color
	if (zwstype .eq. ws_xwindow) then
	   call gqlwk(zwstype,error,max_pline,max_pmark,
	1	max_text,max_fill_area,max_pattern,max_color)
	else
	   error = 0
	   max_pline = 20
	   max_pmark = 20
	   max_text = 20
	   max_fill_area = 20
	   max_pattern = 20
	   max_color = 256
	endif
	return
	end
