        SUBROUTINE SET_ONE_COLOR
        
*       Sets the RGB values for a single color index
*       J Davison JISAO/PMEL/TMAP 11.15.91
*       *jd* 3.10.92 Brought over to Sun
*	*jd* 1.14.94 Mod for background setting in xgks
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn

#ifdef unix
#   ifndef atc_gks
        include        'tmap_pplv11inc/gkspar.inc'
#   endif
	include		'tmap_pplv11inc/cmrd_inc.decl'
        include         'pplv11inc/CMRD.INC'
	include		'tmap_pplv11inc/gkscm1_inc.decl'
        include         'pplv11inc/GKSCM1.INC'
	include		'tmap_pplv11inc/gkscm2.cmn'
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
	      call gscr (wsid,i,val(red),val(green),val(blue))
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
	call gscr (wsid,ndx,val(red),val(green),val(blue))
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
	   call gscr (wsid,ndx,val(red),val(green),val(blue))
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
        call guwk (wsid,gpostp)
#endif

        return
	end


