	SUBROUTINE SETUP_COLOR_TABLE (zwsid,ndx_lo,range)
	
*	Sets the color map between ndx_lo and ndx_hi -- GKS FORTRAN binding
*	J Davison JISAO/PMEL/TMAP 4.29.88/6.15.88

*	Completely rewritten J Davison 7.27.89
*	Mod JD 4.18.90 allowing only one shade level

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		zwsid,ndx_lo,range,ndx,t_cl,cl_0
	integer		ndx_hi,ndx_lop1,ndx_him1

 	real		del_lvl,lvl
	real		r_inc,g_inc,b_inc,r_val,g_val,b_val

******************************************************************************

	ndx_hi		= ndx_lo + range - 1
	ndx_lop1	= ndx_lo + 1
	ndx_him1	= ndx_hi - 1

	if (range .eq. 2) goto 100
	if (range .le. 1) goto 101
	cl_0	= 1
	lvl	= 0.0
	del_lvl	= 100.0/real (range - 1)

*	SET ALL COLORS IN SPECTRUM EXCEPT FIRST AND LAST
	do 50 ndx = ndx_lop1,ndx_him1
	   lvl = lvl + del_lvl

*	   LOCATE THIS LEVEL IN THE SPECIFIED SPECTRUM LEVELS
	   do 10 t_cl = cl_0,ncl-1
	      if (lvl .lt. slvl(t_cl+1)) then
		 cl_0 = t_cl
		 goto 20
	      end if
10	   end do

*	   FIND THE R,G,B VALUES TO SET THIS LEVEL
20	   r_inc	= (sr(cl_0+1) - sr(cl_0))/(slvl(cl_0+1) - slvl(cl_0))
	   g_inc	= (sg(cl_0+1) - sg(cl_0))/(slvl(cl_0+1) - slvl(cl_0))
	   b_inc	= (sb(cl_0+1) - sb(cl_0))/(slvl(cl_0+1) - slvl(cl_0))

	   r_val	= sr(cl_0) + r_inc*(lvl - slvl(cl_0))
	   g_val	= sg(cl_0) + g_inc*(lvl - slvl(cl_0))
	   b_val	= sb(cl_0) + b_inc*(lvl - slvl(cl_0))

*	   NOW SET THE COLOR OF THIS INDEX/LEVEL
	   call gscr (zwsid,ndx,r_val,g_val,b_val)
50	end do

*	SET R,G,B FOR FIRST AND LAST LEVELS
100	call gscr (zwsid,ndx_hi,sr(ncl),sg(ncl),sb(ncl))
101	call gscr (zwsid,ndx_lo,sr(1),sg(1),sb(1))

	return
	end
