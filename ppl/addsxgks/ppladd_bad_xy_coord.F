	LOGICAL FUNCTION BAD_XY_COORD (i,j,xs,py,ndx)

*	Screens 2D plot points where the coordinates of the grid cell are
*	invalid

*	*sh* 4/97 - extracted from "get_ndx" which previously checked both
*	for bad data values and for bad coordinates. With curvilinear coord
*	translations included in the logic these two functions needed to be
*	separated

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

	real		xs(nx+1),py(4)
	real            get_ndx

********************************************************************************

*	      CHECK AGAINST 'LIMITS' VALUES AND FIND COLOR INDEX
	      if((xle.and.xs(i).le.cmxle) 	.or.
     .		 (xeq.and.xs(i).eq.cmxeq) 	.or.
     .		 (xge.and.xs(i+1).ge.cmxge) 	.or.
     .		 (yle.and.py(1).le.cmyle) 	.or.
     .		 (yeq.and.py(1).eq.cmyeq) 	.or.
     .		 (yge.and.py(2).ge.cmyge) ) THEN

	         get_ndx = bad_val
	         BAD_XY_COORD = .TRUE.

	      ELSE

		 BAD_XY_COORD = .FALSE.

	      end if

	      return
	end
