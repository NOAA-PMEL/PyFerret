	SUBROUTINE SET_WINDOW_SIZE (wsid,xmin,xmax,ymin,ymax)

*	Routine to handle resizing windows -- uses GSWKVP for DEC GKS and 
*	GUESC -5304 in ATC GKS.  ATC doesn't resize the window with GSWKVP.
*	J Davison 3.24.92
*	Mod 8.27.93 to handle XGKS resizing
*       Mod 3.15.94 *jd* to call X event processing 
*	*sh* 3.5.97 Mod to support metafile-only (no X) output through a
*		cuckoo's egg implantation of meta_wsid in place of wsid


#ifdef unix
#   ifndef atc_gks
        include         'tmap_pplv11inc/gkspar.inc'
#   endif
        include         'tmap_pplv11inc/ws_types.cmn'
	include         'tmap_pplv11inc/gkscm2.cmn'
#else
        include         'sys$library:gksdefs.bnd'
        include         'tmap_pplv11inc:ws_types.inc'
	include         'tmap_pplv11inc:gkscm2.cmn'
#endif

	integer		wsid,ierr,conid,wstype
	real		xmin,xmax,ymin,ymax,xsize,ysize,xfactor,yfactor

        integer 	ia(3),lidr,lodr 
        real 		ra(1)

        character*80 	datarec(1), odr(1), str(1)

	LOGICAL		ITS_BATCH_GRAPHICS, batch

*******************************************************************************
* initialize batch mode
	batch = ITS_BATCH_GRAPHICS()

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
	call resize_xgks_window (wsid,xmax-xmin,ymax-ymin)
c       Also need to resize the metafile window, if it is open *js* 8.97
	if (meta_actv .and. .not. meta_clsplt) then
	   call resize_xgks_window (meta_wsid, xmax-xmin,ymax-ymin)
	end if
#else
	call gswkvp (wsid,xmin,xmax,ymin,ymax)
#endif

1000	return
	end

