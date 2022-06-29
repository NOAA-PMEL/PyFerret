	INTEGER FUNCTION DO_4D_STD(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					boxes	)

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
* call DO_4D_TRANS with the appropriate transformation

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V695 *acm* 10/15 Fix #2319: @STD as a multi-D transform
* V702 *sh* 11/16 mods for trac #2428 - calculation in gathered fragments
*                 pass address of buffer area, where the components needed to
*                 finalize the 1-pass weighted variance calculation are accumulated 
* v767 6/22 *acm* See issue 112: Use the INTEGER*8 routine CGRID_SIZE8

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, com(*), res(*), boxes(*)

* internal variable declarations:
	INTEGER*8 CGRID_SIZE8, size

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* get size of result array to locate buffer area just beyond
	size = CGRID_SIZE8(res_cx)

* pass the transform to be performed
	CALL DO_4D_TRANS(   trans_4d_std,
     .			    com, com_mr, com_cx,
     .			    res, res_mr, res_cx,
     .			    boxes,
     .			    res(size+1), res(2*size+1) )

	DO_4D_STD = ferr_ok

	RETURN
	END
