	SUBROUTINE SET_WINDOW_SIZE (wsid,xmin,xmax,ymin,ymax)

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
*	Routine to handle resizing windows -- uses GSWKVP for DEC GKS and 
*	GUESC -5304 in ATC GKS.  ATC doesn't resize the window with GSWKVP.
*	J Davison 3.24.92
*	Mod 8.27.93 to handle XGKS resizing
*       Mod 3.15.94 *jd* to call X event processing 
*	*sh* 3.5.97 Mod to support metafile-only (no X) output through a
*		cuckoo's egg implantation of meta_wsid in place of wsid
*      *acm* 8/27/04 New returns from resize_xgks_window; pixx,pixy which are
*                    the size of the window in pixels.  Save these in common 
*                    gkscm2.cmn, so that in gif and batch modes we can return 
*                    them in symbols PPL$XPIXEL and PPL$YPIXEL.
*
#   ifndef atc_gks
        include         'gkspar.inc'
#   endif
        include         'ws_types.cmn'
	include         'gkscm2.cmn'


	integer		wsid,ierr,conid,wstype
	real		xmin,xmax,ymin,ymax,xsize,ysize,xfactor,yfactor

        integer 	ia(3),lidr,lodr 
        real 		ra(1)

        character*80 	datarec(1), odr(1), str(1)

	LOGICAL		ITS_BATCH_GRAPHICS, batch

*******************************************************************************
* initialize batch mode
	batch = ITS_BATCH_GRAPHICS()
        pixx = -1
        pixy = -1

#ifdef atc_gks
	
*	Use the escape -5304 from ATC to resize Xwindowse.  Units are in meters
*       -- to be converted to pixels (raster units).

	call gqwkc (wsid,ierr,conid,wstype)

	if (wstype/100 .eq. ws_xwindow/100) then 
*	   FIND CONVERSION FACTOR FROM PIXELS TO MM FOR THIS MONITOR
           call guesc307 (wstype,ierr,xfactor,yfactor)
	   if (ierr .ne. 0) goto 1000

  	   xsize = xmax - xmin
	   ysize = ymax - ymin

	   ia(1) = wsid
	   ia(2) = xsize * 1000.0 / xfactor
	   ia(3) = ysize * 1000.0 / yfactor

           call gprec (3, ia, 0, ra, 0, 0, str, 80, ierr, lidr, datarec)
           call guesc (-5304, lidr, datarec, 80, lodr, odr)
	else 
	   call gswkvp (wsid,xmin,xmax,ymin,ymax)
	endif
#elif xgks
	call resize_xgks_window (wsid,xmax-xmin,ymax-ymin,pixx,pixy)
c       Also need to resize the metafile window, if it is open *js* 8.97
	if (meta_actv .and. .not. meta_clsplt) then
	   call resize_xgks_window (meta_wsid, xmax-xmin,ymax-ymin,
     .             pixx,pixy)
	end if
#else
	call gswkvp (wsid,xmin,xmax,ymin,ymax)
#endif

1000	return
	end

