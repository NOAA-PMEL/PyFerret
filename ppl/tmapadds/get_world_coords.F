	SUBROUTINE GET_WORLD_COORDS (w,v,wc)

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
*	Finds world coordinates for setting viewport clipping
*    ACM 1/24/03  based on get_user_coords

#ifdef unix
	include		'axis_inc.decl'
	include		'AXIS.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
#endif

	real	 	w(4),v(4),vc(4),ndc(4),wc(4),uc(4)
	real		nax,nay,nbx,nby,wax,way,wbx,wby,uax,uay,ubx,uby
	real		xinv,yinv

c  from get_user_coords:  want to use wc below for clipping region.

*	GET VIEWPORT COORDS OF PLOT ACTIVE REGION CORNERS
	nax = (v(2) - v(1)) / (w(2) - w(1))
	nbx = v(1) - w(1)*nax
	nay = (v(4) - v(3)) / (w(4) - w(3))
	nby = v(3) - w(3)*nay
	
	vc(1) = nax* xorg*1000.0  + nbx
	vc(2) = nax*(xorg + xlen)*1000.0 + nbx
	vc(3) = nay* yorg*1000.0  + nby
	vc(4) = nay*(yorg + ylen)*1000.0 + nby

*	FIND APPROPRIATE CORNERS TO CLIP
	ndc(1) = vc(1)	
	ndc(2) = vc(2)
	ndc(3) = vc(3)
	ndc(4) = vc(4)

*	WHAT WORLD COORDINATES ARE THESE?
	wax = (w(2) - w(1)) / (v(2) - v(1))
	wbx = w(1) - v(1)*wax
	way = (w(4) - w(3)) / (v(4) - v(3))
	wby = w(3) - v(3)*way
	
	wc(1) = wax*ndc(1) + wbx
	wc(2) = wax*ndc(2) + wbx
	wc(3) = way*ndc(3) + wby
	wc(4) = way*ndc(4) + wby

	return
	end
