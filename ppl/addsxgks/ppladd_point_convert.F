	SUBROUTINE POINT_CONVERT (xno_uzr,yno_uzr,xuzr,yuzr)

*	Converts a point (x,y) in /nouser coordinates to /user
*	J Davison 10.17.89

#ifdef unix
	include		'tmap_pplv11inc/plt_inc.decl'
	include		'pplv11inc/PLT.INC'
	include		'tmap_pplv11inc/axis_inc.decl'
	include		'pplv11inc/AXIS.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_plt.inc'
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
#endif

	real		xno_uzr,yno_uzr,xuzr,yuzr,xf,yf

********************************************************************************

	xf = (xhi - xlo)/xlen
	yf = (yhi - ylo)/ylen

	xuzr = xno_uzr*xf + xlo
	yuzr = yno_uzr*yf + ylo
                                                                               
	if (itypex .gt. 1) xuzr = 10.0**xuzr
	if (itypey .gt. 1) yuzr = 10.0**yuzr

	return
	end
