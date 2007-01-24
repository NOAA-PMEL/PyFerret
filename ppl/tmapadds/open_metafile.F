	SUBROUTINE OPEN_METAFILE

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
*       2.20.98 *kob* - moved query_xgks_wkstn to its own routine.  causing
*                       problems when trying to link gksm2ps
* V602  1/07 *acm*  true batch mode for metafiles; flag meta_batch
*         distinguises this case from other batch modes.

	include 'pltcom_dat.decl'! with declarations    
	include 'PLTCOM.DAT'! with declarations    
	include 'gkscm1_inc.decl'! with declarations    
	include 'GKSCM1.INC'! with declarations    
	include 'gkscm2.cmn'! with declarations    
	include 'ppl_in_ferret.cmn'
	include 'ws_types.cmn'

        integer       park_stat


#ifdef atc_gks
	include        'gkspar.inc'
#else
  	include        'gkspar.inc'
#endif

	integer 	ndx,type,error,len,tm_lenstr
	real		red,green,blue

	integer		errind,errnr,fctid
	integer         theWsType


	LOGICAL ITS_META_GRAPHICS, ITS_BATCH_GRAPHICS
        LOGICAL its_meta, batch, meta_batch

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

        its_meta = ITS_META_GRAPHICS()
        batch = ITS_BATCH_GRAPHICS()
        meta_batch = batch .AND. its_meta

#ifdef xgks
	if (meta_actv .OR. meta_batch) then
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

