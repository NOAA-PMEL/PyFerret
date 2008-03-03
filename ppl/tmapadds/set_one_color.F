        SUBROUTINE SET_ONE_COLOR
        
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
*       Sets the RGB values for a single color index
*       J Davison JISAO/PMEL/TMAP 11.15.91
*       *jd* 3.10.92 Brought over to Sun
*	*jd* 1.14.94 Mod for background setting in xgks
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn
* V610  *acm* 3/08 Fixes for metafile batch mode

#ifdef unix
#   ifndef atc_gks
        include        'gkspar.inc'
#   endif
	include		'cmrd_inc.decl'
        include         'CMRD.INC'
	include		'gkscm1_inc.decl'
        include         'GKSCM1.INC'
	include		'gkscm2.cmn'
#else
	include		'tmap_pplv11inc:tmap_cmrd.inc'
	include		'tmap_pplv11inc:tmap_gkscm1.inc'
	include		'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

	integer		ndx,red,green,blue,i,err
        integer         max_pline,max_pmark,max_text,max_fill_area
        integer         max_pattern,max_color
	real		val(3)

	parameter	(red = 1,green = 2,blue = 3)	

#ifdef xgks
	integer		black, white
	data		black/0/,white/1/
#endif

******************************************************************************

*	IF NO ARGS, RESET COLORS
	if (m .eq. 0) then
           call query_xgks_wkstn (wstype,err,max_pline,max_pmark,
     .              max_text,max_fill_area,max_pattern,max_color)
	   if (err .ne. 0) goto 1000

#ifndef xgks
	   do 5 i = 0,6
#else
           do 5 i = 1,6
#endif
	      call gqpcr (wstype,i,err,val(red),val(green),val(blue))
	      if (err .ne. 0) goto 5
	      if (meta_actv) call gscr (wsid,i,val(red),val(green),val(blue))
	      if (meta_actv .and. .not. meta_clsplt) 
     .		call gscr (meta_wsid,i,val(red),val(green),val(blue))
5	   continue
	   goto 1000
	end if

*	ELSE SET COLOR P(1) to RGB: P(2), P(3), P(4)
	if (m .ne. 4) goto 1000
	
	if (p(1) .ge. 0.0 .and. p(1) .le. 255) then
	   ndx = ifix (p(1))
	else
	   goto 1000
	end if

	do 10 i = 1,3
	   if (p(i+1) .ge. 0.0 .and. p(i+1) .le. 100.0) then
	      val(i) = p(i+1)/100.0
	   else
	      goto 1000
	   end if
10	continue

#ifndef xgks
	if (meta_actv) call gscr (wsid,ndx,val(red),val(green),val(blue))
#else
*	FOR X, PERMIT ONLY WHITE OR BLACK BACKGROUNDS FOR NOW
	if (ndx .eq. 0) then
	   if (val(red)   .eq. 0.0 .and.
     .	       val(green) .eq. 0.0 .and. 
     .	       val(blue)  .eq. 0.0) then 
	       call set_background (wsid,black)
	   else
	       call set_background (wsid,white)
	   endif
	else
	   if (meta_actv) 
     .        call gscr (wsid,ndx,val(red),val(green),val(blue))
	endif
#endif

	if (meta_actv .and. .not. meta_clsplt) then
#ifndef xgks
           call gscr (meta_wsid,ndx,val(red),val(green),val(blue))
#else
*	   FOR X, PLACE NO INDEX 0 IN METAFILE, AND INDEX 1 CANNOT BE WHITE
	   if (ndx .eq. 0) goto 1000
	   if (ndx .eq. 1) then
	      if (val(red)   .eq. 1.0 .and.
     .		  val(green) .eq. 1.0 .and. 
     .		  val(blue)  .eq. 1.0) then 
                  call gscr (meta_wsid,ndx,0.0,0.0,0.0)
	      else
                 call gscr (meta_wsid,ndx,val(red),val(green),val(blue))
	      endif
	   else
              call gscr (meta_wsid,ndx,val(red),val(green),val(blue))
	   endif
#endif
	endif
 1000   continue

#ifndef atc_gks
        if (meta_actv) call guwk (wsid,gpostp)
#endif

        return
	end


