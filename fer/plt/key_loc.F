	SUBROUTINE KEY_LOC( overlay, xpos, ypos, xsize, ysize,
     .			    lab_ht, style, justify )

* find a location, size and style in which to put the next key

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  1/2/90

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/plot_setup.parm'
	include	'ferret_cmn/xplot_setup.cmn'
	include 'tmap_pplv11inc/axis_inc.decl'		! axis lengths
	include 'pplv11inc/AXIS.INC'		! axis lengths
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'		! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'		! axis lengths
#endif

* calling argument declarations
	LOGICAL overlay
	INTEGER	style, justify
	REAL	xpos, ypos, xsize, ysize, lab_ht

* internal variable declarations
	INTEGER	colmn

	IF ( overlay ) THEN
* is there room for this overlay label at the lower left ?
	   ysize = ovkey_yspace
	   ypos  = ovkey_yorigin - novkeys_on*ovkey_yspace
	   IF ( ABS(ypos).GT.yorg .AND. novkeys_ll.EQ.0 ) THEN
	      ovkey_ll_full = .TRUE.
	      novkeys_ll = novkeys_on
	   ENDIF
	   IF ( ovkey_ll_full ) THEN
* ... no - try to put the label within the plot rectangle at the left
	      xsize  = ovkey_xspace * 0.8
	      xpos   = 0.1
	      lab_ht = ovkey_labht  * 0.8
	      ysize = ovkey_yspace  * 0.8	      
	      ypos = 0.1 + (novkeys_on-novkeys_ll)*ovkey_yspace*0.8
	      IF (ABS(ypos+lab_ht) .GT. ylen ) xpos = -999.	! off screen
	   ELSE
* ... yes - it will fit at the lower left
	      xsize  = ovkey_xspace
	      xpos   = ovkey_xorigin
	      lab_ht = ovkey_labht
	   ENDIF
	   novkeys_on = novkeys_on + 1

	ELSE
* locate a regular (non-overlay) key below the axis
	   xsize = key_xsize
	   ysize = key_ysize
	   lab_ht = key_labht
	   colmn = MOD( nkeys_on,key_ncolmn )	! from 0 to n-1
	   xpos = key_xorigin + colmn*key_xspace
	   ypos = key_yorigin - INT(nkeys_on/key_ncolmn)*key_yspace
	   nkeys_on = nkeys_on + 1
	ENDIF

* style and justify aren't really used in FERRET version 2.00
	style   = ppl_sidebyside
	justify = ppl_left_just	

	RETURN
	END
