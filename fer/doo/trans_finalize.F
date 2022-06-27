	SUBROUTINE TRANS_FINALIZE( idim, arg,
     .				   res, res_mr, res_cx )

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
* finalize a transform that contains accumulated values
* e.g. average =  WIGHTED_SUM(values) divided by SUM(weights)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702 11/16 *sh* for enhancement ticket 2428: split/gather on compressed axes
* v767 6/22 *acm* See issue 112: The total size could need INTEGER*8


        IMPLICIT NONE
* calling argument declarations:
	INTEGER	idim, res_mr, res_cx
	REAL	arg, res(*)

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* internal variable declarations:
	INTEGER	trans
	INTEGER*8 CGRID_SIZE8, size

* initiailize
	trans = cx_trans( idim, res_cx )
	size = CGRID_SIZE8(res_cx)

	IF ( trans .EQ. trans_average
     .  .OR. trans .EQ. trans_4d_ave   
     .  .OR. trans .EQ. trans_integ_def
     .  .OR. trans .EQ. trans_4d_int_def  ) THEN
* ... diagnostic mode output: " final --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	   IF ( mode_diagnostic ) CALL DIAG_OP
     .			( 'final', isact_class_trans, res_cx, idim )
	   CALL FINALIZE_AVE_INT(trans, size,
     .			         res_mr, res, res(size+1) )

	ELSEIF ( trans .EQ. trans_sum
     .      .OR. trans .EQ. trans_4d_sum
     .      .OR. trans .EQ. trans_max
     .      .OR. trans .EQ. trans_min   ) THEN

	   CALL FINALIZE_SUMS_COUNTS(trans, size,
     .			         res_mr, res, res(size+1) )

	ELSEIF ( trans .EQ. trans_variance
     .      .OR. trans .EQ. trans_4d_var
     .	    .OR. trans .EQ. trans_std_dev
     .      .OR. trans .EQ. trans_4d_std ) THEN

	   IF ( mode_diagnostic ) CALL DIAG_OP
     .			( 'final', isact_class_trans, res_cx, idim )
	   CALL FINALIZE_VARIANCE(trans, size,
     .			          res_mr, res, res(size+1), res(2*size+1) )


	ELSE
 
* trans_good_pt, trans_4d_good_pt, trans_bad_pt, trans_4d_bad_pt are simple 
* counts, and do not need a finalization step.

* all other transformations do not require initialization
*      trans_locate - potentially could, but would need added machinery during
*                     split gather since it is locating a result that lies
*                     *between* two points


	   CONTINUE   ! TBD

	ENDIF

	RETURN
	END
