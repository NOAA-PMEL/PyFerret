      SUBROUTINE QUERY_DISPLAY_SIZE (wstype,ierr,imeter,x,y,ix,iy)

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
