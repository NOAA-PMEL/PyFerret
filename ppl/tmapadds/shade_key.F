        SUBROUTINE SHADE_KEY
        
*       Sets the options for SHADE key -- size,location,etc
*       J Davison JISAO/PMEL/TMAP 8.24.88

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef unix
	include		'tmap_pplv11inc/cmrd_inc.decl'
	include		'pplv11inc/CMRD.INC'
        include         'tmap_pplv11inc/axis_inc.decl'
        include         'pplv11inc/AXIS.INC'
        include         'tmap_pplv11inc/plt_inc.decl'
        include         'pplv11inc/PLT.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_cmrd.inc'
        INCLUDE         'tmap_pplv11inc:tmap_axis.inc'
        INCLUDE         'tmap_pplv11inc:tmap_plt.inc'
#endif

*******************************************************************************

	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.0.0) 
     .	do_key = .false.
	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.1.0) 
     .	do_key = .true.

	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.0.0) 
     .	vertical_key = .false.
	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.1.0) 
     .	vertical_key = .true.

	if (m .ge. 3 .and. iflg(3) .ne. 0) klab_size = p(3)
	if (m .ge. 4 .and. iflg(4) .ne. 0) klab_incr = ifix (p(4))
	if (m .ge. 5 .and. iflg(5) .ne. 0) klab_dig  = ifix (p(5))
	if (m .ge. 6 .and. iflg(6) .ne. 0) klab_len  = ifix (p(6))

	if (m .ge. 10 .and. iflg(7) .ne. 0 .and. iflg(8) .ne. 0 
     .	.and. iflg(9) .ne. 0 .and. iflg(10) .ne. 0) then
	   kuser_loc = .true.
	   kx_lo = p(7) * 1000.0
	   kx_hi = p(8) * 1000.0
	   ky_lo = p(9) * 1000.0
	   ky_hi = p(10)* 1000.0
        else
	   kuser_loc = .false.
        end if

	return
	end
