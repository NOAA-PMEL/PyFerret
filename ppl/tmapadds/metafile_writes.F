*	PPLADD_METAFILE_WRITES

*	Write private item data records to metafile
*	J Davison JISAO/PMEL/TMAP
*	10.26.89
*	Mod *JD* 4.22.91 Make all entry points into subroutines and
*	add support for new metafile translator
*       Mod *JD* 5.9.91 Work around GKD3D bug wrt private items
*	Above mod taken out when GKS3D bug is fixed late 91.
*	*jd* 2.25.92 Modify for ATC gks -- binary GKSM files require packed rec
*       *jd* 24.7.92 Mod to change back as before for DEC GKS -- doesn't work
*	*jd* 10.5.92 Mod to add info re: protect_used_colors for ATC.
*	*jd* 10.12.92 Mod to add fill area ndx for ATC who stupidly have but
*	    20 area bundles and consequently have a useless area bundle context

	SUBROUTINE META_CLEAR (control_flag)

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef atc_gks
	integer		control_flag,ix(4)
	real 		x(4)
#else
       integer          control_flag
       character*8      kar_control_flag(1)
#endif 

***************************************************************************

C	Mod for linux *jd* 12.96 change kar_control_flag to array for 
C	consistency with WRITE_ITEM call to gwitm.

#ifdef atc_gks
	ix(1) = control_flag
	call write_item (ix,x,item_gclrwk)
#else
        write (unit=kar_control_flag(1),fmt='(i8)') control_flag
        call gwitm (meta_wsid,item_gclrwk,8,1,kar_control_flag)
#endif
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE META_WS_XFORM (x_wn,y_wn,big_x_vp,big_y_vp)

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

        real            x_wn,y_wn,big_x_vp,big_y_vp,x_vp,y_vp

#ifdef atc_gks
	integer		ix(4)
	real 		x(4)
#else
        character*8     kar_x_wn,kar_y_wn,kar_x_vp,kar_y_vp
        character*32    kar_ws_xform(1)
#endif                                                                    

C	Mod for linux *jd* 12.96 change kar_ws_xform to array for 
C	consistency with WRITE_ITEM call to gwitm.

***************************************************************************

*	CONVERT VIEWPORT SIZE TO INCHES FROM MILLI-INCHES
	x_vp = big_x_vp/1000.0
	y_vp = big_y_vp/1000.0

#ifdef atc_gks
	x(1) = x_wn
	x(2) = y_wn
	x(3) = x_vp
	x(4) = y_vp

	call write_item (ix,x,item_ws_xform)
#else
        x_vp = big_x_vp/1000.0
        y_vp = big_y_vp/1000.0

        write (unit=kar_x_wn,fmt='(f8.4)') x_wn
        write (unit=kar_y_wn,fmt='(f8.4)') y_wn

        write (unit=kar_x_vp,fmt='(f8.4)') x_vp
        write (unit=kar_y_vp,fmt='(f8.4)') y_vp

        kar_ws_xform(1) = kar_x_wn//kar_y_wn//kar_x_vp//kar_y_vp
        call gwitm (meta_wsid,item_ws_xform,32,1,kar_ws_xform)
#endif

	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE META_SHADE_SET

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
	include		'tmap_pplv11inc/shade_vars.cmn'
        include         'tmap_pplv11inc/gkscm1_inc.decl'
        include         'pplv11inc/GKSCM1.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
	INCLUDE		'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef atc_gks
        integer         i,ix(4)
	real		x(4)
#else
        integer         i
        character*8     kar_shd_levels,kar_ncl,kar_loco
        character*8     kar_ndx,kar_slvl,kar_sr,kar_sg,kar_sb
        character*24    kar_ndx_range(1)
        character*40    kar_spectrum(1)
#endif

C	Mod for linux *jd* 12.96 change kar_ndx_range, kar_spectrum to array 
C	for consistency with WRITE_ITEM call to gwitm.


***************************************************************************

*	WRITE # SHADE LEVELS & # COLOR LEVELS
#ifdef atc_gks
	ix(1) = shd_levels
	ix(2) = ncl
	ix(3) = lo_color_ndx(wsid)

	call write_item (ix,x,item_ndx_range)
#else
        write (unit=kar_shd_levels,fmt='(i8)') shd_levels
        write (unit=kar_ncl,fmt='(i8)') ncl
        write (unit=kar_loco,fmt='(i8)') lo_color_ndx(wsid)

        kar_ndx_range(1) = kar_shd_levels//kar_ncl//kar_loco
        call gwitm (meta_wsid,item_ndx_range,24,1,kar_ndx_range)
#endif

*	WRITE OUT COLOR REPRESENTATIONS
	do 10 i = 1,ncl
#ifdef atc_gks
	   ix(1) = i

	   x(1) = slvl(i)
	   x(2) = sr(i)
	   x(3) = sg(i)
	   x(4) = sb(i)

	   call write_item (ix,x,item_spectrum)
#else
           write (unit=kar_ndx,fmt='(i8)') i
           write (unit=kar_slvl,fmt='(f8.4)') slvl(i)
           write (unit=kar_sr,fmt='(f8.4)') sr(i)
           write (unit=kar_sg,fmt='(f8.4)') sg(i)
           write (unit=kar_sb,fmt='(f8.4)') sb(i)

           kar_spectrum(1) = kar_ndx//kar_slvl//kar_sr//kar_sg//kar_sb
           call gwitm (meta_wsid,item_spectrum,40,1,kar_spectrum)
#endif
10	continue

	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE META_PROTECT

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
	include		'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
	INCLUDE		'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef atc_gks
        integer         ix(4)
	real		x(4)
#else
        integer         i
        character*8     kar_protect(1)
#endif

C	Mod for linux *jd* 12.96 change kar_protect to array for 
C	consistency with WRITE_ITEM call to gwitm.

***************************************************************************

*	WRITE OUT PROTECT_USED_COLORS INFO AS ATC DOESN'T USE FILL AREA
*	BUNDLES AND COLOR INDICES MUST BE PROPERLY RE-CREATED

#ifdef atc_gks
	if (protect_used_colors) then
	   ix(1) = 1
	else
	   ix(1) = 0
	endif

	call write_item (ix,x,item_protect)
#else
	if (protect_used_colors) then
	   i = 1
	else
	   i = 0
	endif

        write (unit=kar_protect(1),fmt='(i8)') i
        call gwitm (meta_wsid,item_protect,8,1,kar_protect)
#endif
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE META_FILL_NDX (fill_ndx)

	integer		fill_ndx

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef atc_gks
        integer         ix(4)
	real		x(4)

***************************************************************************

*	WRITE OUT FILL AREA INDEX CURRENTLY IN USE
	ix(1) = fill_ndx

	call write_item (ix,x,item_fill_ndx)
#endif
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE META_VERSION

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef unix
        include 	'tmap_pplv11inc/pplp_ver_date.cmn'
#else
        INCLUDE 	'TMAP_PPLV11INC:PPLP_VER_DATE.CMN'
#endif

#ifdef atc_gks
	integer		ix(4)
	real 		x(4)
#else
       character*13     kar_rev(1)
#endif 

***************************************************************************

#ifdef atc_gks
	read (unit=pplp_ver,fmt='(1x,f4.2)') x(1)
	read (unit=pplp_make_date,fmt='(i2)') ix(1)
	read (unit=pplp_make_date,fmt='(3x,i2)') ix(2)
	read (unit=pplp_make_date,fmt='(6x,i2)') ix(3)

	call write_item (ix,x,item_pplpver)
#else
        kar_rev(1) = pplp_ver // pplp_make_date
        call gwitm (meta_wsid,item_pplpver,13,1,kar_rev)
#endif
	return
	end

C	Mod for linux *jd* 12.96 change kar_rev to array for 
C	consistency with WRITE_ITEM call to gwitm.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE WRITE_ITEM (ix,x,item_type)

*	Converts data to packed record format and writes out item to metafile
*	J Davison 2.25.92

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

C	Moved variable initialization into data statement for linux *jd* 12.96

	integer		ix(4),item_type
	real 		x(4)

	integer		lstr(1),ierr,ldr
	character*4	str(1)
	character*80	datrec(10)

	data		lstr/4/,str/'str!'/


*******************************************************************************

	call gprec (4,ix,4,x,1,lstr,str,10,ierr,ldr,datrec)
	call gwitm (meta_wsid,item_type,80,ldr,datrec)

	return
	end





