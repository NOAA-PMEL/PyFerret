	INTEGER FUNCTION GET_NDX (data,i,j,ndx)

*	Gets bundle index for fill area calls
*	J Davison 5.17.90
*	Mod *jd* 10.1.92 to use 1 fill index                    

* 	Mod *sh* 4/97 -- removed checks on valid coordinates from this routine
*		to a separate new routine, BAD_XY_COORD

#ifdef unix
	include		'tmap_pplv11inc/parampl5_dat.decl'
	include		'pplv11inc/PARAMPL5.DAT'
	include		'tmap_pplv11inc/miss_inc.decl'
	include		'pplv11inc/MISS.INC'
        include         'tmap_pplv11inc/hd_inc.decl'
        include         'pplv11inc/HD.INC'
	include		'tmap_pplv11inc/cont_inc.decl'
	include		'pplv11inc/CONT.INC'
	include		'tmap_pplv11inc/shade_vars.cmn'

#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
	INCLUDE		'tmap_pplv11inc:tmap_miss.inc'
        INCLUDE         'tmap_pplv11inc:tmap_hd.inc'
	INCLUDE		'tmap_pplv11inc:tmap_cont.inc'
	INCLUDE		'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		i,j,ndx

	real		data(nx,ny)

********************************************************************************

*	      CHECK AGAINST 'LIMITS' VALUES AND FIND COLOR INDEX
	      if(
     .		 (zle.and.data(i,j).le.cmzle) 	.or.
     .		 (zeq.and.data(i,j).eq.cmzeq) 	.or.
     .		 (zge.and.data(i,j).ge.cmzge)) then

	         get_ndx = bad_val
	      else if (nlev2 .eq. 1) then
		 if (data(i,j) .eq. zlev(1)) then
		    get_ndx = 2
		 else
		    get_ndx = bad_val
		 endif
	      else             
	         call locator (zlev,1,nlev2,data(i,j),ndx)

	         if (ndx .eq. nlev2) then
	            get_ndx = bad_val
	         else
	            get_ndx = ndx + 1
	         end if
	      end if

	return
	end
