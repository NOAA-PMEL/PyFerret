	LOGICAL FUNCTION BAD_XY_COORD (i,j,xs,py,ndx)

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
*	Screens 2D plot points where the coordinates of the grid cell are
*	invalid

*	*sh* 4/97 - extracted from "get_ndx" which previously checked both
*	for bad data values and for bad coordinates. With curvilinear coord
*	translations included in the logic these two functions needed to be
*	separated

#ifdef unix
	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'miss_inc.decl'
	include		'MISS.INC'
        include         'hd_inc.decl'
        include         'HD.INC'
	include		'cont_inc.decl'
	include		'CONT.INC'
	include		'shade_vars.cmn'

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
