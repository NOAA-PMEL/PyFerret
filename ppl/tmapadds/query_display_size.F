      SUBROUTINE QUERY_DISPLAY_SIZE (wstype,ierr,imeter,x,y,ix,iy)

*     With use if ATC GKS modification to display size query GQDSP
*     is needed. This routine handles both ATC and DEC GKS queries.
*     J Davison 3.24.92
*     Mod 8.27.93 *jd* to deal w xgks

#ifdef unix
#   ifndef atc_gks
        include         'tmap_pplv11inc/gkspar.inc'
#   endif
        include         'tmap_pplv11inc/ws_types.cmn'
#else
        include         'sys$library:gksdefs.bnd'
        include         'tmap_pplv11inc:ws_types.inc'
#endif

      integer     wstype,ierr,imeter,ix,iy
      real        dx_in,dy_in,x,y,xfactor,yfactor

******************************************************************************

#ifdef atc_gks
*     Mod 1 will assume 13x10 inch Xmonitors.  ATC does not restrict windows
*     to fit inside display.  Also GKS query returns size of window, not
*     the display.  

      dx_in = 13.0
      dy_in = 10.0

      if (wstype/100 .eq. ws_xwindow/100) then
*        RETURN VALUES IN METERS AND PIXELS
         x = dx_in * 0.0254
         y = dy_in * 0.0254

*        HOW MANY MM/PIXEL
         call guesc307 (wstype,ierr,xfactor,yfactor)

         ix = dx_in * 25.4 / xfactor
         iy = dy_in * 25.4 / yfactor

         imeter = 0
         ierr = 0
      else
*        TAKE WHAT WE GET FOR TEKS ETC
         call gqdsp (wstype,ierr,imeter,x,y,ix,iy)
      endif
#elif xgks
	 ierr = 0
	 imeter = 0
	 x = 0.36
	 y = 0.25
	 ix = 1280
	 iy = 1024
#else
*     USING DEC GKS USE NORMAL QUERY
      call gqdsp (wstype,ierr,imeter,x,y,ix,iy)
#endif
      
      return
      end
