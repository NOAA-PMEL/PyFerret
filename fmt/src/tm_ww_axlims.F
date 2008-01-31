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

	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include	'xtm_grid.cmn_text'

* calling argument declarations
	INTEGER axis
	REAL*8 lo, hi

* internal variable declarations
	INTEGER	isub1, llen

	IF ( axis .LT. 0 .OR. axis .GT. line_ceiling ) THEN
	  lo = unspecified_val8
	  hi = unspecified_val8
	ELSEIF ( line_regular(axis) ) THEN
	  lo = line_start(axis) - 0.5D0*line_delta(axis)
	  hi = lo + line_dim(axis)*line_delta(axis)
	ELSE
	  isub1 = line_subsc1(axis)
	  llen = line_dim(axis)
	  hi = line_mem(isub1 + 2*llen)
	  lo = line_mem(isub1 +   llen)
	ENDIF

	RETURN
	END
