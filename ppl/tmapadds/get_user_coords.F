	SUBROUTINE GET_USER_COORDS (w,v,ndc,uc,xinv,yinv)

*	Finds user coordinates given world coordinates and ancillary ppl info.
*	J Davison 5.17.90

#ifdef unix
	include		'tmap_pplv11inc/axis_inc.decl'
	include		'pplv11inc/AXIS.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
#endif

	real	 	w(4),v(4),vc(4),ndc(4),wc(4),uc(4)
	real		nax,nay,nbx,nby,wax,way,wbx,wby,uax,uay,ubx,uby
	real		xinv,yinv

********************************************************************************

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

*	WHAT ARE THE USER COORDS OF THESE POINTS?
	uax = (xhi - xlo) / (xlen*1000.0)
	ubx = xlo - (xorg*1000.0) * uax
	uay = (yhi - ylo) / (ylen*1000.0)
	uby = ylo - (yorg*1000.0) * uay
	
	uc(1) = uax*wc(1) + ubx
	uc(2) = uax*wc(2) + ubx
	uc(3) = uay*wc(3) + uby
	uc(4) = uay*wc(4) + uby

*	ARE AXES INVERTED?
	if (uc(1) .gt. uc(2)) then
	   xinv = -1.0
	else
	   xinv =  1.0
	endif

	if (uc(3) .gt. uc(4)) then
	   yinv = -1.0
	else
	   yinv =  1.0
	endif

	return
	end
