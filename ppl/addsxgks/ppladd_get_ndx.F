	INTEGER FUNCTION GET_NDX (datum,ndx)

*	Gets bundle index for fill area calls
*	J Davison 5.17.90
*	Mod *jd* 10.1.92 to use 1 fill index                    

* 	Mod *sh* 4/97 -- removed checks on valid coordinates from this routine
*		to a separate new routine, BAD_XY_COORD

* 	Mod *sh* 11/97 -- changed calling args by eliminating "i,j" so this
*		routine can be used for 1D (polygon fill) as well as 2D
*		Also: bug in this code: if there is no valid
*	        index for datum then "bad_val" was returned. But bad_val is
*		a REAL!!   Now return -999 instead (quick hack).
*		Also had to add explicit range checks since whatever logic
*		made rng chks mysteriously unnecessary using REAL bad_flag has
*		changed

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

	integer		ndx

	real		datum

********************************************************************************

*	      CHECK AGAINST 'LIMITS' VALUES AND FIND COLOR INDEX
	      if(
     .		 (zle.and.datum.le.cmzle) 	.or.
     .		 (zeq.and.datum.eq.cmzeq) 	.or.
     .		 (zge.and.datum.ge.cmzge)) then

	         get_ndx = -999
* ... 12/4/97 added range checks
	      else if (datum .lt. zlev(1)) then
	         get_ndx = -999
	      else if (datum .gt. zlev(nlev2)) then
	         get_ndx = -999
* ... 12/4/97

	      else if (nlev2 .eq. 1) then
		 if (datum .eq. zlev(1)) then
		    get_ndx = 2
		 else
		    get_ndx = -999
		 endif
	      else             
	         call locator (zlev,1,nlev2,datum,ndx)

	         if (ndx .eq. nlev2) then
	            get_ndx = -999
	         else
	            get_ndx = ndx + 1
	         end if
	      end if

	return
	end
