	SUBROUTINE TM_WW_AXLIMS (axis, lo, hi)

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
* Return the lowest and highest points along the span of an axis
* For a regular axis this will be 1/2 delta below/above the 1st/Nth coordinates
* Modulo is ignored in this calculation

* V541 *sh* 2/02
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes.

	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include	'xtm_grid.cmn_text'

* calling argument declarations
	INTEGER axis
	REAL*8 lo, hi

* internal variable declarations
	LOGICAL ITSA_TRUEMONTH_AXIS
	INTEGER	llen, iiaxis
	REAL	TM_WORLD, GET_LINE_COORD

	IF ( axis .LT. 0 .OR. axis .GT. line_ceiling ) THEN
	  lo = unspecified_val8
	  hi = unspecified_val8
c	ELSEIF ( ITSA_TRUEMONTH_AXIS(axis) ) THEN
c	  lo = TM_WORLD (1, axis, box_lo_lim )
c	  hi = TM_WORLD (line_dim(axis), axis, box_hi_lim )
	ELSEIF ( line_regular(axis) ) THEN
	  lo = line_start(axis) - 0.5D0*line_delta(axis)
	  hi = lo + line_dim(axis)*line_delta(axis)
	ELSE
	  iiaxis = axis
	  IF (line_parent(axis) .NE. 0) iiaxis = line_parent(axis)
	  llen = line_dim(iiaxis)
	  
	  lo = GET_LINE_COORD (lineedg(iiaxis)%ptr, 1)
	  hi = GET_LINE_COORD (lineedg(iiaxis)%ptr, llen+1)

	ENDIF

	RETURN
	END
