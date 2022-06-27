	SUBROUTINE FINALIZE_AVE_INT( trans, size,
     .				     res_mr, res, weights )

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
* finalize the result of average or definite integral based up accumulated
* WIGHTED_SUM(values) and SUM(weights)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702 *sh* 11/16 this routine performs the finalization step that was
*		  previously a part of DO_AVE_INT
* V72  *acm* 6/17 Ticket 2539: Do not set res to bad-value for ave, din.
*                 Weights of zero or non-zero mark whether data is found.
* v767 6/22 *acm* See issue 112: The total size could need INTEGER*8

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
!	include	'xcontext.cmn'
	include	'xvariables.cmn'
!	include	'xmem_subsc.cmn'
!	include 'xunits.cmn_text'
!	external xunits_data
!	include 'xtm_grid.cmn_text'
!	external xgt_grid_data

* calling argument declarations:
	INTEGER*8 size
	INTEGER	trans, res_mr
	REAL    res(size), weights(size)

* internal variable declarations:
	LOGICAL	average
	INTEGER	i 
	REAL	bad_res

* --- end of introductory code ---

* initialize
	average = trans .EQ. trans_average
     .       .OR. trans .EQ. trans_4d_ave
	bad_res = mr_bad_data( res_mr )

* finalize (luv it!)
	DO i = 1, size
	   IF (weights(i) .EQ. 0.0D0) THEN  
	      res(i) = bad_res			! no input values to average
	   ELSEIF (average) THEN
	      res(i) = res(i)/weights(i)	! average
	   ENDIF
	ENDDO

	RETURN
	END
