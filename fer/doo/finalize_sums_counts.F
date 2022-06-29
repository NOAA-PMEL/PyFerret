	SUBROUTINE FINALIZE_SUMS_COUNTS( trans, size,
     .				     res_mr, res, counts )

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
* finalize the result of Sums or Counts (NGD, NBD).

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V71 *acm* 12/16 this routine performs the finalization step that was
*		  previously a part of DO_SUM, DO_MIN, DO_MAX, DO_NGOOD, etc.
* v767 6/22 *acm* See issue 112: The total size could need INTEGER*8

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER*8 size
	INTEGER	trans, res_mr
	REAL    res(size), counts(size)

* internal variable declarations:
	INTEGER	i
	REAL	bad_res

* --- end of introductory code ---

* initialize
	bad_res = mr_bad_data( res_mr )

* res(i) already contains the accumulated sum or count.
* counts(i) contains the number of valid data.

	DO i = 1, size
	   IF (counts(i) .EQ. 0.0D0) THEN
	      res(i) = bad_res			! no valid input values 
	   ENDIF
	ENDDO

	RETURN
	END
