	SUBROUTINE DO_AUX_VAR_REGRID( memory, axis_list, naxis, aux_has_cell_pts,
     .				      src,  msrc,  src_cx,
     .				      dst,  mdst,  dst_cx,
     .				      aux1, maux1, aux1_cx,
     .				      aux2, maux2, aux2_cx,
     .				      aux3, maux3, aux3_cx,
     .				      status )

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
* perform a dispatch of regridding to the rountine appropriate
! for the dimensionailty

* v685 4/13-4/14 *sh*
* v689+ 3/15 *sh* added support for T axis T-F orthoganalization

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	LOGICAL	aux_has_cell_pts
	INTEGER	naxis, axis_list(naxis), status,
     .		msrc, mdst, maux1, maux2, maux3,
     .		src_cx, dst_cx, aux1_cx, aux2_cx, aux3_cx
	REAL	memory( mem_blk_size, max_mem_blks )
        REAL    src(*), dst(*), aux1(*), aux2(*), aux3(*)

* internal variable declarations
	INTEGER	axis, indmin_dst, indmax_dst

	IF ( naxis .EQ. 1) THEN

* initialize
	   axis = axis_list(1)

	   IF (axis .EQ. z_dim) THEN
* if this is a "1pt false regridding", in which there is no new target axis
* such as results from Z=0 or Z=100:200@ave, then the cx_ss values are
* -999:-999.  To avoid accidental "bad value" interpretations we replace those
	      IF (cx_aux_stat(axis,dst_cx)
     .	     .EQ. paux_stat_false_1pt_regrid) THEN
	         indmin_dst = 1   ! replacing -999
	         indmax_dst = 1   ! replacing -999
	      ELSE
	         indmin_dst = cx_lo_ss(dst_cx,axis)
	         indmax_dst = cx_hi_ss(dst_cx,axis)
	      ENDIF

* do the 1D Z axis regridding
	      CALL DO_AUX_VAR_REGRID_1D_Z(
     .			axis, aux_has_cell_pts,
     .			src,  msrc,  src_cx,
     .			dst,  mdst,  dst_cx,
     .			aux1, maux1, aux1_cx,
     .			cx_lo_ss(src_cx,axis),cx_hi_ss(src_cx,axis),
     .			indmin_dst,           indmax_dst              )

	
	   ELSEIF (axis .EQ. t_dim) THEN

* do the 1D T axis T-F orthoganalization
	      CALL DO_AUX_VAR_REGRID_1D_T(
     .			axis,
     .			src,  msrc,  src_cx,
     .			dst,  mdst,  dst_cx,
     .			aux1, maux1, aux1_cx,
     .			cx_lo_ss(dst_cx,t_dim),cx_hi_ss(dst_cx,t_dim) )

	   ELSE
	       STOP 'mystery 1D regrid should not be happening'
	   ENDIF

	ELSEIF (naxis .EQ. 2) THEN
	   IF (axis_list(1).EQ.t_dim .AND. axis_list(2).EQ.f_dim) THEN

* do the 1D T axis T-F orthoganalization
	      CALL DO_AUX_VAR_REGRID_2D_TF(
     .			src,  msrc,  src_cx,
     .			dst,  mdst,  dst_cx,
     .			aux1, maux1, aux1_cx,
     .			cx_lo_ss(dst_cx,t_dim),cx_hi_ss(dst_cx,t_dim) )
	   ELSE
	       STOP 'mystery 2D regrid should not be happening'
	   ENDIF
	ELSE
	       STOP 'mystery N-D regrid should not be happening'
	ENDIF

* success always -- all possible error conditions were anticipated
	RETURN
	END
