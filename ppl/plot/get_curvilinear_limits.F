	SUBROUTINE GET_CURVILINEAR_LIMITS( xmin, xmax, ymin, ymax )

* Copy the plot limits, stored in curvilinear common, to the arguments

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 3/97
*
* 9/98 - *kob* - this routine was originaly located in fer/plt but because
*                of linking difficulties w/ the newly organized source code
*                it was deemed more apporpriate in this directory

#ifdef unix
	include 'ferret_cmn/xcurvilinear.cmn'
#else
#endif

* calling argument declarations:
	REAL xmin, xmax, ymin, ymax

* internal variable declarations:

	xmin = wwxmin
	xmax = wwxmax
	ymin = wwymin
	ymax = wwymax

	RETURN
	END
