	SUBROUTINE DO_AUX_REGRID_LINE_LIN ( axis,
     .					 src_line,
     .					 kmin_src, kmax_src,
     .					 indices, coef,
     .					 kmin_dst, kmax_dst,
     .					 kdlo_ok,  kdhi_ok,
     .					 bad_src,  bad_dst, ascending,
     .				         dst_line             )

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
* regrid a single line using linear interpolation
* using the index pointers and coefficients provided

* v690 11/13 *sh*

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
     .		indices (kmin_dst:kmax_dst)
        REAL    src_line(kmin_src:kmax_src),
     .		dst_line(kmin_dst:kmax_dst),
     .		coef    (kmin_dst:kmax_dst),
     .		bad_src, bad_dst

* internal variable declarations:
        INTEGER ndx_lo, k, kinc1
        REAL    frac

* initialize
	IF (ascending) THEN
	   kinc1 = 1
	ELSE
	   kinc1 = -1
	ENDIF

* loop performing linear interpolations
	DO 1000 k = kdlo_ok, kdhi_ok   ! dest Z axis

	   ndx_lo = indices(k)
	   frac   = coef(k)

           IF ( frac .LT. 0.0 ) THEN
              dst_line(k) = bad_dst
           ELSEIF ( src_line(ndx_lo) .EQ. bad_src ) THEN
              dst_line(k) = bad_dst
           ELSEIF ( frac .EQ. 1.0 ) THEN
              dst_line(k) = src_line(ndx_lo)
           ELSEIF ( src_line(ndx_lo+kinc1) .EQ. bad_src ) THEN
              dst_line(k) = bad_dst
           ELSE
* ... interpolate data from com1 field to get result
              dst_line(k) = 
     .                 frac  * src_line(ndx_lo)
     .            + (1-frac) * src_line(ndx_lo+kinc1)
           ENDIF
 1000   CONTINUE

	RETURN
	END
