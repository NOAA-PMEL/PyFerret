      SUBROUTINE NEW_GKS_WINDOW (ws_id,xname,status)

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
*     windows on workstation screen.  Modified from GKS-3D FORTRAN Binding
*     Manual example 19-1.  For VMS simply open the workstation.
*     J Davison 5.13.91
*     *jd* 3.26.92  Modified to accommodate ATC GKS as well as DEC
*     *jd* 12.21.93 Modified to accommodate xgks


      integer         ws_id,status
      character*(*)   xname
      character*2     kwsid

#ifdef xgks
      character*8     window_name

      include         'tmap_pplv11inc/gkspar.inc' 
      
      write (unit=kwsid,fmt='(i2)') ws_id

      window_name = 'FERRET_' // kwsid(2:2)
      call gesspn (window_name,8)
      call gopwk (ws_id,6,4)

*     SET DEFERRAL STATE TO GENERATE OUTPUT AS SOON AS POSSIBLE
      call gsds (ws_id,gasap,gsuppd)
#else
#ifdef atc_gks
      integer         con_id

      character*80    conid_options 
      character*80    conid_opt/' -nc64 -ox 2 -u 2'/,conid_opt_def
#else
      include         'tmap_pplv11inc/gkspar.inc' 

      integer         i,fctid,mldr,mlodr,odr
      integer         il,ia,rl,ra,sl,lstr(1),errind,ldr,lidr,lodr

      character*80    idr,str(1)
#endif

#ifdef unix
      include 'tmap_pplv11inc/gkscm1_inc.decl'
      include 'pplv11inc/GKSCM1.INC'
      include 'tmap_pplv11inc/ws_types.cmn'
#else
      INCLUDE 'TMAP_PPLV11INC:GKSCM1_INC.DECL'
      INCLUDE 'PPLV11INC:GKSCM1.INC'
      INCLUDE 'tmap_pplv11inc:ws_types.inc'
#endif

*******************************************************************************

#ifdef atc_gks
      con_id = ws_id
      write (unit=kwsid,fmt='(i2)') ws_id
      conid_opt_def = '-l FERRET'//kwsid//conid_opt
#   ifdef unix
      call getenv ('atcGKSconid',conid_options)
#   else
*     GET LOGICAL WITH TRNLNM
#   endif
      if (conid_options .eq. ' ') conid_options = conid_opt_def

      if (wstype/100 .eq. ws_xwindow/100) then
         call guesc050 (con_id,conid_options)
         call guesc005 (con_id)
      endif
      call gopwk (ws_id,con_id,wstype)
#else
#   ifdef unix
*     Define the conid string by use of pack record (GPREC) and the FORTAN 
*     binding GESC_CONID escape function.

      il = 0
      ia = 0
      rl = 0
      ra = 0
      sl = 1
      mldr = 10
      lstr(1) = len (xname)
      str(1)  = xname
      call gprec (il,ia,rl,ra,sl,lstr,str,mldr,errind,ldr,idr)

      fctid = gecon
      lidr  = 28
      mlodr = 0
      lodr  = 0
      call gesc (fctid,lidr,idr,mlodr,lodr,odr)

      call gopwk (ws_id,0,0)
#   else
      call GKS$OPEN_WS (ws_id,xname,0)
#   endif
#endif

#endif
      status = 3   ! Meaningless but may be used later
      return 
      end
      

