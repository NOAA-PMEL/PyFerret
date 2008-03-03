	SUBROUTINE SET_GKS_METAFILE

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
*	J Davison JISAO/PMEL/TMAP
*	3.15.89

*	For GKS plots (pltype = 3 or 4) captures WS type and whether to open a
*	GKS metafile, or not.  Opens GKS if not open, and activates/deactivates
*	metafile WS.

*       *JD* 4.22.91 No need for metafile wstype w new metafile translator
*	*jd* 3.10.92 Mod to handle ATC GKS methods
*	*jd* 10.7.92 Mod to use wstype default (GKS3Dwstype) in DEC
*	*jd* 2.24.93 Mod to add PPLP version numbers in metafile
*	*jd* 8.10.93 Mod to add XGKS options for WS types
*	*jd* 10.26.93 Mod to eliminate version number from xgks
*       *jd* 12.8.93 Mod to put line color rep info into metafile for xgks
*	*sh* 3.4.97 Mod to support metafile-only (no X) output through a
*		cuckoo's egg implantation of meta_wsid in place of wsid
*       *js* 7.25.97 Added 'C' callable save_metafile_name to save string
*               in variable 'meta_file'
*       *acm* 2.08 Added 'C' callable get_metafile_name to retrieve string
*               in variable 'meta_file' and return it.
* v552 *acm* 3/03 Longer strings.
* V608 *acm*10/07  true batch mode for metafiles; flag meta_batch
*         distinguises this case from other batch modes.
* V610  *acm* 3/08 Fixes for metafile batch mode

	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'cmrdl_inc.decl'
	include 'CMRDL.INC'
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'
	include 'gkscm2.cmn'
	include 'ws_types.cmn'

	integer		beg,end,defws,ndx,type,error
	real		red,green,blue
	character*2048	str
	character*5	kdefws

	LOGICAL ITS_META_GRAPHICS, ITS_BATCH_GRAPHICS
        LOGICAL its_meta, batch, meta_batch

#ifdef unix
        logical         xwindows
#else
        integer         trnlnm,xwindows
	character*256	name
	external	trnlnm
#endif

*************************************************************************

*	DETERMINE IF WINDOWING IS X OR NOT
#ifdef unix
        xwindows = .true.
#else
	xwindows = trnlnm ( 'DECW$LOGICAL_NAMES', 'DECW$SYSTEM_DEFAULTS',
     .						 name, namelen )
#endif

* Metafile output only?
        its_meta = ITS_META_GRAPHICS()
        batch = ITS_BATCH_GRAPHICS()
        meta_batch = batch .AND. its_meta

*	MASSAGE COMMAND STRING OF FORM "WS=xxx,META"
	call upnsquish (label,str,end)	 	 	   	

*	OPEN GKS AND WS IF NOT ALREADY OPEN
	if (gksopn) goto 10

*	DEFINE DEFAULT WS TYPE
#ifdef unix
#   ifdef atc_gks
	call getenv ('atcGKSwstype',kdefws)
#   elif xgks
	call getenv ('XGKSwstype',kdefws)
#   else
	call getenv ('GKS3Dwstype',kdefws)
#   endif
# else
*	USE TRNLNM CALL FOR VMS
#endif

#ifdef atc_gks
	if (kdefws .ne. ' ') then
	   read (unit=kdefws,fmt='(i5)') defws
	else
	   defws = ws_xwindow
	endif
#elif xgks
	if (kdefws .ne. ' ') then
	   read (unit=kdefws,fmt='(i5)') defws
	else
	   defws = ws_xwindow
	endif

#else
        defws = 0
#endif

*	EXTRACT WS INFO FROM PLTYPE STRING ARGUMENT
	beg = index (str,'WS')

*	USE DEFAULT IF WS TYPE NOT SPECIFIED BELOW
	if (beg .eq. 0) then
	   wstype = defws

	else
	   if (index (str(beg+3:beg+6),'VSII') .ne. 0) then
	      if (xwindows) then
		 wstype = ws_xwindow
	      else 
		 wstype = ws_uis
	      end if
	   else if (index (str(beg+3:beg+9),'TEK4014') .ne. 0) then
	      wstype = ws_tek4014
	   else if (index (str(beg+3:beg+9),'TEK4107') .ne. 0) then
	      wstype = ws_tek4107
	   end if
	end if
	call open_gks_ws

*	EXTRACT METAFILE CMD FROM PLTYPE STRING ARGUMENT
10	beg = index (str,'META')

	if (beg .ne. 0) then
*	   METAFILE SHOULD BE OPEN AND ACTIVE

	   if (.not. meta_open) call open_metafile
	   if (.not. meta_open) return

	   if (.not.meta_actv) then
	      if (.not. meta_clsplt) then 
		 call gacwk (meta_wsid)
#ifndef xgks
		 call meta_version
#endif
#ifdef xgks
*               ALLOW NO WHITE LINES IN METAFILE (FOR NOW)
                call gqcr (wsid,1,type,error,red,green,blue)
                if (error .eq. 0) then
                   if (red   .eq. 1.0 .and.
     .                 green .eq. 1.0 .and.
     .                 blue  .eq. 1.0) then
                      call gscr (meta_wsid,1,0.0,0.0,0.0)
                    else
                      call gscr (meta_wsid,1,red,green,blue)
                    endif
                else
                   call gscr (meta_wsid,1,0.0,0.0,0.0)
                endif

		do ndx = 2,6
		   call gqcr (wsid,ndx,type,error,red,green,blue)	
		   if (batch .or. error .eq. 0) then
		      call gscr (meta_wsid,ndx,red,green,blue)
		   else
		      call gscr (meta_wsid,ndx,0.0,0.0,0.0)
		   endif
		end do
#endif
	      endif
! 3/97: although the metafile is in fact opened we are going to make it
! look like the output workstation to all other software layers
c	      IF ( batch .AND. (.NOT.meta_batch)) THEN
	      IF ( batch) THEN
	         meta_actv = .false.
	         meta_open = .false.
	      ELSE
	         meta_actv = .true.
	      ENDIF
	   end if
	else
*	   IF NO "META" IN PLTYPE CMD DEACTIVATE METAFILE FOR NOW IF ACTIVE
	   if (meta_actv) then
	      if (.not. meta_clsplt) call gdawk (meta_wsid)
	      meta_actv = .false.
	   end if
	end if

	return
	end


	subroutine save_metafile_name(mname, clen)
	implicit none
	include 'gkscm2.cmn'
	integer clen
#ifdef sun
	byte mname(clen) 
#else   
	integer*1 mname(clen)
#endif  
	call tm_ctof_strng(mname, meta_file, clen)
	return
	end
	

	subroutine get_metafile_name(filename)
	implicit none
	include 'gkscm2.cmn'
        character*(*) filename

	filename = meta_file
	return
	end
	
