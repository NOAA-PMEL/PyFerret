	SUBROUTINE KEY 

*	Generates color/gray scale key
* 	J Davison JISAO/PMEL/NOAA
*	7.20.88

*	Modified to PPL_KEY to fit into ppl
*	J Davison 8.17.88
*	Mod introducing xmaxx and ymaxx to limit size of key JD 4.11.90
*	Mod to control font thickness JD 8.3.90
*	Mod to support ATC individual fill area reps *jd* 4.2.92
*       Mod to support single level chosen by user 8.25.92 *jd*
*	Mod to protect colors used already *jd* 10.1.92
*	Mod to use new hatch scheme *jd* 10.12.92 - cancelled 10.22.92
*	Mod to use SYMBEL rather than GTX for key labels *jd* 3.4.93
* 	Mod to provide identical endpoint values for a constant field/jd/5.96

#ifdef unix
	include 	'tmap_pplv11inc/parampl5_dat.decl'
	include 	'pplv11inc/PARAMPL5.DAT'
	include		'tmap_pplv11inc/cont_inc.decl'
	include		'pplv11inc/CONT.INC'
        include 	'tmap_pplv11inc/pltcom_dat.decl'
        include 	'pplv11inc/PLTCOM.DAT'
        include 	'tmap_pplv11inc/pltl_inc.decl'
        include 	'pplv11inc/PLTL.INC'
#else
	INCLUDE 	'tmap_pplv11inc:tmap_parampl5.dat'
	INCLUDE		'tmap_pplv11inc:tmap_cont.inc'
        INCLUDE 	'TMAP_PPLV11INC:LINES_INC.DECL'
        INCLUDE 	'PPLV11INC:LINES.INC'
        INCLUDE 	'TMAP_PPLV11INC:PLTCOM_DAT.DECL'
        INCLUDE 	'PPLV11INC:PLTCOM.DAT'
        INCLUDE 	'TMAP_PPLV11INC:PLTL_INC.DECL'
        INCLUDE 	'PPLV11INC:PLTL.INC'
#endif

#ifdef unix
        include         'tmap_pplv11inc/axis_inc.decl'
        include         'pplv11inc/AXIS.INC'
        include         'tmap_pplv11inc/plt_inc.decl'
        include         'pplv11inc/PLT.INC'
        include         'tmap_pplv11inc/pen_inc.decl'
        include         'pplv11inc/PEN.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_axis.inc'
        INCLUDE         'tmap_pplv11inc:tmap_plt.inc'
        INCLUDE         'tmap_pplv11inc:tmap_pen.inc'
#endif

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
        include         'tmap_pplv11inc/gkscm1_inc.decl'
        include         'pplv11inc/GKSCM1.INC'
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

	integer		ndx,str_len,max_len,incr

	real		x_inc,y_inc,px(4),py(4),rem,kpx(5),kpy(5)
	real		xmaxx,ymaxx,vwidth,vheight,karht
	real		xx,yy,xxx,yyy

	character*12	text,tm_fmt
	logical		windof_hold

	external	tm_fmt

	equivalence 	(px(1),kpx(1)),(py(1),kpy(1))

#ifdef unix
	include 	'tmap_pplv11inc/ppl_in_ferret.cmn'	
#else
	INCLUDE 	'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'	
#endif

*******************************************************************************

*	CANCEL WINDOW ON IF SET
	windof_hold = windof
	windof = .false.

	if (vertical_key) goto 1000

*	MAKE A HORIZONTAL KEY
	if (.not. kuser_loc) then
           if (ppl_in_ferret) then
              call get_view_size (vwidth,vheight)
              if (vheight .le. ylen + yorg) goto 2000

              rem = vheight - (ylen + yorg)
	      ymaxx = 0.25*ylen
              kx_lo = 1000.0 *  xorg
              kx_hi = 1000.0 * (xorg + xlen)
              ky_lo = 1000.0 * (yorg + ylen + 0.25*min (rem,ymaxx))
              ky_hi = 1000.0 * min (yorg+ylen+ymaxx,vheight-0.1*rem)
           else
              rem = height - (ylen + yorg)
	      ymaxx = 0.25*ylen
              kx_lo = 1000.0 *  xorg
              kx_hi = 1000.0 * (xorg + xlen)
              ky_lo = 1000.0 * (yorg + ylen + 0.25*min (rem,ymaxx))
              ky_hi = 1000.0 * min (yorg+ylen+ymaxx,height-0.1*rem)
           end if
        end if

	x_inc = (kx_hi - kx_lo)/max (1.0, real (shd_levels))
	y_inc = 0.1 * (ky_hi - ky_lo)

*	GENERATE THE KEY FOR EACH COLOR/SHADE
	max_len = 0
	do 10 ndx = 2, max (2,shd_levels + 1)
	   px(1) = kx_lo + real (ndx - 2) * x_inc
	   px(2) = kx_lo + real (ndx - 2) * x_inc
	   px(3) = kx_lo + real (ndx - 1) * x_inc
	   px(4) = kx_lo + real (ndx - 1) * x_inc

	   py(1) = ky_hi
	   py(2) = ky_hi - 4.0 * y_inc 
	   py(3) = ky_hi - 4.0 * y_inc
	   py(4) = ky_hi

*	   CALL FILL AREA FOR THIS COLOR/GRAY VALUE
	   call set_fill_ndx (ndx)
	   call gfa (4,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   NDX TO PRIVATE ITEM -- forget it 10.22
*	   if (meta_actv .and. .not. area_bundles) call meta_fill_ndx (ndx)

	   kpx(5) = kpx(1)
	   kpy(5) = kpy(1)
	   call gpl (5,kpx,kpy)

	   text = tm_fmt (zlev(ndx),klab_dig,klab_len,str_len)
	   if (max_len .lt. str_len) max_len = str_len	   
10	continue

*	SET CHARACTER HEIGHT IN WORLD COORDINATES, AND INCR INDEX
	if (klab_incr .eq. 0) then
	   incr = shd_levels/25 + 1
	else
	   incr = klab_incr
	end if

	if (klab_size .eq. 0.0) then
	   karht = amin1 (0.08*1000.0,
     .		1.0*x_inc*real(incr)/real(max_len))/1000.0
	else
	   karht = klab_size
	end if

	do 20 ndx = 1,shd_levels+1,incr
*	   WRITE OUT CORRESPONDING VALUE

CC	   FIX HERE FOR CONSTANT FIELD -- SPECIAL CASE		
	   if (shd_levels .eq. 1 .and. ndx .eq. 2) then
	     text = tm_fmt (zlev(1),  klab_dig,klab_len,str_len)
	   else
	     text = tm_fmt (zlev(ndx),klab_dig,klab_len,str_len)
	   end if

	   xx = (kx_lo + x_inc*real(ndx-1))/1000.0
     .		 - karht*real(str_len)/2.0 - xorg
	   yy = (ky_hi - 5.0 * y_inc)/1000.0 - yorg - karht

*	   FILL AND SHADE USE DIFFERENT COORD SYSTEMS -- *jd* 6.4.93
	   if (makep .eq. 'SHADE') then
	      call point_convert (xx,yy,xxx,yyy)
	   else if (makep .eq. 'FILL') then
	      xxx = xx
	      yyy = yy
	   endif

	   call symbel (xxx,yyy,0.0,karht,str_len,text(1:str_len))

20	continue
	goto 2000

*	MAKE A VERTICAL KEY 
1000	if (.not. kuser_loc) then
           if (ppl_in_ferret) then
              call get_view_size (vwidth,vheight)
              if (vwidth .le. xlen + xorg) goto 2000

	      rem = vwidth - (xlen + xorg)
	      xmaxx = 0.25*xlen
              kx_lo = 1000.0 * (xorg + xlen + 0.25*min (rem,xmaxx))
              kx_hi = 1000.0 * min (xorg + xlen + xmaxx,vwidth-0.1*rem)
              ky_lo = 1000.0 *  yorg
              ky_hi = 1000.0 * (yorg + ylen)
           else
	      rem = width - (xlen + xorg)
	      xmaxx = 0.25*xlen
              kx_lo = 1000.0 * (xorg + xlen + 0.25*min (rem,xmaxx))
              kx_hi = 1000.0 * min (xorg + xlen + xmaxx,width-0.1*rem)
              ky_lo = 1000.0 *  yorg
              ky_hi = 1000.0 * (yorg + ylen)
           end if
        end if

	x_inc = 0.1 * (kx_hi - kx_lo)
	y_inc = (ky_hi - ky_lo)/max (1.0, real (shd_levels))

*	GENERATE THE KEY FOR EACH COLOR/SHADE
	max_len = 0
	do 1010 ndx = 2, max (2,shd_levels + 1)
	   px(1) = kx_lo
	   px(2) = kx_lo + 4.0*x_inc
	   px(3) = kx_lo + 4.0*x_inc
	   px(4) = kx_lo

	   py(1) = ky_lo + real (ndx - 2) * y_inc
	   py(2) = ky_lo + real (ndx - 2) * y_inc
	   py(3) = ky_lo + real (ndx - 1) * y_inc
	   py(4) = ky_lo + real (ndx - 1) * y_inc

*	   CALL FILL AREA FOR THIS COLOR/GRAY VALUE
	   call set_fill_ndx (ndx)
	   call gfa (4,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   NDX TO PRIVATE ITEM - forget it 10.22
*	   if (meta_actv .and. .not. area_bundles) call meta_fill_ndx (ndx)

           kpx(5) = kpx(1)
           kpy(5) = kpy(1)
           call gpl (5,kpx,kpy)

	   text = tm_fmt (zlev(ndx),klab_dig,klab_len,str_len)
	   if (max_len .lt. str_len) max_len = str_len	
1010	continue

*	SET CHARACTER HEIGHT IN WORLD COORDINATES, AND INCR INDEX
        if (klab_size .eq. 0.0) then
	   karht = amin1 (0.08*1000.0,360.0*rem/real (max_len))/1000.0
	else
	   karht = klab_size
	end if

	if (klab_incr .eq. 0) then
	   incr = shd_levels/25 + 1
	else
	   incr = klab_incr
	end if

	do 1020 ndx = 1,shd_levels+1,incr
*	   WRITE OUT CORRESPONDING VALUE

CC	   FIX HERE FOR CONSTANT FIELD -- SPECIAL CASE		
	   if (shd_levels .eq. 1 .and. ndx .eq. 2) then
	     text = tm_fmt (zlev(1),  klab_dig,klab_len,str_len)
	   else
	     text = tm_fmt (zlev(ndx),klab_dig,klab_len,str_len)
	   end if

	   xx = (kx_lo + 5.0*x_inc)/1000.0 - xorg 
	   yy = (ky_lo + y_inc*real (ndx-1))/1000.0 - yorg - 0.5*karht

*	   FILL AND SHADE USE DIFFERENT COORD SYSTEMS -- *jd* 6.4.93
	   if (makep .eq. 'SHADE') then
	      call point_convert (xx,yy,xxx,yyy)
	   else if (makep .eq. 'FILL') then
	      xxx = xx
	      yyy = yy
	   endif

	   call symbel (xxx,yyy,0.0,karht,str_len,text(1:str_len))

1020	continue

2000	windof = windof_hold
	return
	end






