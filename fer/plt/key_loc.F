	SUBROUTINE KEY_LOC( overlay, xpos, ypos, xsize, ysize,
     .			    lab_ht, style, justify )

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
