	SUBROUTINE SET_DEFAULT_COLORS (zwsid,zwstype,ndx_lo,range)
	
*	Sets the color map between ndx_lo and ndx_hi to ws default
*	J Davison JISAO/PMEL/TMAP 8.1.89

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		zwsid,ndx_lo,range,ndx_hi,zwstype,err,ndx
	real		r_val,g_val,b_val

********************************************************************************

	ndx_hi	= ndx_lo + range - 1

*	SET ALL COLORS TO THE DEFAULT
	do 50 ndx = ndx_lo,ndx_hi

*	   WHAT IS THE DEFAULT COLOR FOR EACH NDX?
	   call gqpcr (zwstype,ndx,err,r_val,g_val,b_val)
	   if (err .ne. 0) goto 50

*	   SET THE COLOR OF THIS NDX
	   call gscr (zwsid,ndx,r_val,g_val,b_val)
50	end do

	return
	end
