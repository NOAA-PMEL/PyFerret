	SUBROUTINE SAY_INTERRUPTED

* write a text string "**INTERRUPTED**" on the plot

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/24/89

#ifdef unix
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
#endif

	REAL	x, y, ang, ht, wid, strwid
	CHARACTER string*32
!	REAL	SYMWID		! used once to determine string width

	DATA	ang   / 0.0  /
	DATA	ht    / 0.25 /
	DATA	string/ '@TI** INTERRUPTED **' /
	DATA	strwid/ 3.595 /	! result of "SYMWID( ht, 17, string(4:) )"

	wid = strwid * ((xhi-xlo)/xlen)
	x   = (xlo+xhi-wid)*.5 
	y   = (ylo+yhi)*.5

	CALL SYMBEL( x, y, ang, ht, 20, string )

	RETURN
	END
