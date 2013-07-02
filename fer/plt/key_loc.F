	SUBROUTINE KEY_LOC( overlay, xpos, ypos, xsize, ysize,
     .			    lab_ht, style, justify, string )

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
* find a location, size and style in which to put the next key

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  1/2/90
* v552 *acm* 3/03 multi-line strings in key labels.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* PyFr *kms*  7/13 Moved key_xspace0, key_yspace0, key_labht0, key_xorigin, 
*                  key_yorigin, key_xwhite_space, key_ywhite_space, 
*                  key_line_frac, ovkey_space, ovkey_yspace, ovkey_labht,
*                  ovkey_xorigin, ovkey_yorigin, ul_labht0, ul_yspace0,
*                  ul_xorigin, and ul_yorigin to plot_setup.parm,
*                  since they are used as parameters, and prefixed with 
*                  dflt_ those that need to be scaled by pplscale.

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'plot_setup.parm'
	include	'xplot_setup.cmn'
	include 'axis_inc.decl'		! axis lengths
	include 'AXIS.INC'		! axis lengths
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'labels_inc.decl'
        include 'LABELS.INC'            ! multi-line string

* calling argument declarations
	LOGICAL overlay
	INTEGER	style, justify
	REAL	xpos, ypos, xsize, ysize, lab_ht
        CHARACTER*(*) string

* internal variable declarations
	INTEGER	colmn, numlines
	REAL dylen

	dylen = ylen

* See if the label has more than one line

        CALL tm_break_lines (string, lnbeg, lnend, numlines)

	IF ( overlay ) THEN

* is there room for this overlay label at the lower left ?
* test the location of the last line of this key label

	   ysize = dflt_ovkey_yspace*pplscale
c	   ypos  = ovkey_yorigin - novkeys_on*ovkey_yspace
	   ypos  = dflt_ovkey_yorigin*pplscale - 
     .             (novkeys_on+numlines-1)*dflt_ovkey_yspace*pplscale
	   IF ( ABS(ypos).GT.yorg .AND. novkeys_ll.EQ.0 ) THEN
	      ovkey_ll_full = .TRUE.
	      novkeys_ll = novkeys_on
	   ENDIF

* actual y position
	   ypos  = dflt_ovkey_yorigin*pplscale - 
     .             novkeys_on*dflt_ovkey_yspace*pplscale

	   IF ( ovkey_ll_full ) THEN
* ... no - try to put the label within the plot rectangle at the left
	      xsize  = dflt_ovkey_xspace*pplscale * 0.8
	      xpos   = 0.1*pplscale
	      lab_ht = dflt_ovkey_labht*pplscale  * 0.8
	      ysize  = dflt_ovkey_yspace*pplscale * 0.8
	      ypos   = 0.1*pplscale + 
     .               (numlines-1+novkeys_on-novkeys_ll) *
     .                 dflt_ovkey_yspace*pplscale * 0.8
	      IF (ABS(ypos+lab_ht) .GT. dylen ) xpos = -999.	! off screen
	   ELSE
* ... yes - it will fit at the lower left
	      xsize  = dflt_ovkey_xspace*pplscale
	      xpos   = dflt_ovkey_xorigin*pplscale
	      lab_ht = dflt_ovkey_labht*pplscale
	   ENDIF
	   novkeys_on = novkeys_on + numlines

	ELSE
* locate a regular (non-overlay) key below the axis
	   xsize = key_xsize
	   ysize = key_ysize
	   lab_ht = key_labht
	   colmn = MOD( nkeys_on,key_ncolmn )	! from 0 to n-1
	   xpos = dflt_key_xorigin*pplscale + colmn*key_xspace
	   ypos = dflt_key_yorigin*pplscale - INT(nkeys_on/key_ncolmn)*key_yspace
	   nkeys_on = nkeys_on + numlines
	ENDIF

* style and justify aren't really used in FERRET version 2.00
	style   = ppl_sidebyside
	justify = ppl_left_just	

	RETURN
	END
