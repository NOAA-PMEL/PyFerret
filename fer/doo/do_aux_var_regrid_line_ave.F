	SUBROUTINE DO_AUX_REGRID_LINE_AVE ( axis,
     .					 src_line, pos_line,
     .					 kmin_src, kmax_src,
     .					 ndx_dst, coef_dst,
     .					 kmin_dst, kmax_dst,
     .					 kdlo_ok,  kdhi_ok,
     .					 bad_src,  bad_dst, ascending,
     .					 dst_line    )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* regrid a single line along the regrided axis.  For each destination point
* regrid using the weighted average of values lying entirely or partly
* in the cell

* the algorithm will weight the input values based upon viewing the
* coordinates found in the auxiliary coordinate variable as a piecewise-linear
* function.  For each destination grid cell the algorithm will compute the
* integral, and divide by its length.

* Thus that calculation is one of adding up the areas of a
* series of trapezoids from the source data over a coordinate length defined
* by the limits of each target grid cell.

* For any given cell the trapezopids at the edges of the cell may be truncated,
* as the destination cell boundaries may not lie directly on the coordinate
* values given in the auxiliary variable.

* sample expression:  temp[gz(depth)=zax]
* variable naming convention:
*     pos_line - position (coordinate) values (e.g. depth)
*     src_line - values on their original grid (e.g. temp)
*     dst_line - values regridded to target grid

*  The incoming index pointers tell us the source data indices that lie
*  at or immediatly below each target grid cell boundary.
*  If there are N destination grid points there will be N+1
*  cell boundaries and N+1 incoming indices, positions and coefficients. The
*  coefficient values are non-unity, when the target cell boundary coordinate
*  lies in between the coordinate values from the auxiliary coordinate array
*  The sense of the coefficient is that the value at this destination location
*  should be computed as 
*                  interp_val = coef*src(k) + (1-coef)*src(k+1)
* 
* v690 11/13 *sh* - based upon DO_AUX_VAR_REGRID_LIN

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	LOGICAL	ascending
	INTEGER	axis, kmin_src,kmax_src, kmin_dst, kmax_dst, kdlo_ok, kdhi_ok,
     .		ndx_dst (kmin_dst:kmax_dst+1)
        REAL    src_line(kmin_src:kmax_src),
     .		pos_line(kmin_src:kmax_src),
     .		coef_dst(kmin_dst:kmax_dst+1),
     .		dst_line(kmin_dst:kmax_dst),
     .		bad_src, bad_dst

* internal variable declarations:
	LOGICAL	trunc
        INTEGER ndx_lo, ksrc, kdst, kend, kstart, kinc1
        REAL    area, span, width, frac, pos_lo, pos_hi, src_lo, src_hi

* initialize
	IF (ascending) THEN
	   kinc1 = 1
	ELSE
	   kinc1 = -1
	ENDIF

* set up the lower bound of the first trapezoid (possibly truncated)
 110	ksrc   = ndx_dst (kdlo_ok)
	pos_hi = pos_line(ksrc)
	src_hi = src_line(ksrc)
	frac   = coef_dst(kdlo_ok)
	trunc  = frac .NE. 1.0
	IF (trunc) THEN
* make the initial truncated trapezoid behave like a full trapezoid by  
* replacing its lower limit with the proper linear interpolation
	   src_hi = frac*src_hi + (1-frac)*src_line(ksrc+kinc1)
	   pos_hi = frac*pos_hi + (1-frac)*pos_line(ksrc+kinc1)
	   trunc  = .FALSE.
	ENDIF
	kstart = ksrc + kinc1   ! ksrc always points to the upper index

* loop over destination cells
* each dst cell is represented by two points in the dst arrays
*    the k   point is the index below or at the start of the cell
*    the k+1 point is the index below or at the end   of the cell
	DO 1000 kdst = kdlo_ok, kdhi_ok

	   area = 0.0
	   span = 0.0

* set up low lim of next dest cell from high lim of preceding
	   pos_lo = pos_hi
	   src_lo = src_hi

* loop over the full src trapezoids that contribute to current dst cell
	   kend = ndx_dst(kdst+1)
	   DO 500 ksrc = kstart, kend, kinc1

* upper end of trapezoid in the src line
	      pos_hi  = pos_line(ksrc)
	      src_hi  = src_line(ksrc)

* add a full trapezoid 
	      width = pos_hi - pos_lo
	      span  = span + width
	      area  = area + width * 0.5*(src_hi+src_lo)

* prepare for next trapezoid in the src line
	      pos_lo = pos_hi
	      src_lo = src_hi	    
 500	   CONTINUE

* handle partial trapezoid at upper bound of destination cell
	   frac = coef_dst(kdst+1)
	   trunc = frac .NE. 1.0
	   IF (trunc) THEN
* add the lower part of the truncated trapezoid.
* the upper part is handled as part of the next destination cell
	      pos_hi = frac*pos_line(kend) + (1-frac)*pos_line(kend+kinc1)
	      src_hi = frac*src_line(kend) + (1-frac)*src_line(kend+kinc1)
	      width = pos_hi - pos_lo
	      span  = span + width
	      area  = area + width * 0.5*(src_hi+src_lo)
	   ENDIF

* store the weighted average of trapezoids contributing to dst cell
	   dst_line(kdst) = area/span

* prepare for next src trapezoid contributing to this dst
	   kstart = kend + kinc1

 1000	CONTINUE

	RETURN
	END
