	SUBROUTINE TRANS_INIT_VALUE( idim, cx,
     .			             need_init, init_val, nbuffer)

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
* determine the value to use when initializing the output buffer in preparation
* for performing a transformation
* @AVE and @DIN, for example, require the result array to be initialized to
* zero so that sums can be accumulated
* @SUM needs to count up the number of good-values - an all-bad chunk contributes
* 0 to the sum; we need to know if there's no valid data over the accumulated sum.
* @AVE requires nbuffer=2, because it accumulates both weighted sum of values
* and sum of weights (to be divided in the finalize stage), whereas @DIN
* requires only nbuffer = 1


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702 11/16 *sh* for enhancement ticket 2428: split/gather on compressed axes

* calling argument declarations:
	LOGICAL need_init
	INTEGER idim, cx, nbuffer
	REAL	init_val

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'

* internal variable declarations:
	INTEGER	trans

* internal parameter declaration
        REAL            too_small
        PARAMETER     ( too_small = -1.E33 )
        REAL            too_large
        PARAMETER     ( too_large = 1.E33 )

* --- end of introductory code ---
* initiailize
	trans = cx_trans( idim, cx )

	IF ( trans .EQ. trans_average
     .  .OR. trans .EQ. trans_4d_ave   
     .  .OR. trans .EQ. trans_integ_def
     .  .OR. trans .EQ. trans_4d_int_def  ) THEN
* ... integration accumulates weights, too, so a zero accumulated weight
*     results in a missing value, instead of a zero
	   need_init = .TRUE.
	   init_val = 0.0
	   nbuffer = 2
	ELSEIF (trans .EQ. trans_4d_good_pt
     .      .OR. trans .EQ. trans_bad_pt
     .      .OR. trans .EQ. trans_4d_bad_pt
     .      .OR. trans .EQ. trans_sum
     .      .OR. trans .EQ. trans_4d_sum
     .      .OR. trans .EQ. trans_good_pt
     .      .OR. trans .EQ. trans_4d_good_pt
     .      .OR. trans .EQ. trans_bad_pt
     .      .OR. trans .EQ. trans_4d_bad_pt   ) THEN
	   need_init = .TRUE.
	   init_val = 0.0
	   nbuffer = 1
	ELSEIF ( trans .EQ. trans_variance
     .      .OR. trans .EQ. trans_4d_var
     .	    .OR. trans .EQ. trans_std_dev
     .      .OR. trans .EQ. trans_4d_std ) THEN
	   need_init = .TRUE.
	   init_val = 0.0
	   nbuffer = 3
	ELSEIF ( trans .EQ. trans_max ) THEN
	   need_init = .TRUE.
	   init_val = too_small
	   nbuffer = 1
	ELSEIF ( trans .EQ. trans_min ) THEN
	   need_init = .TRUE.
	   init_val = too_large
	   nbuffer = 1
	ELSE
* all other transformations do not require initialization
*      trans_locate - potentially could, but would need added machinery during
*                     split gather since it is locating a result that lies
*                     *between* two points
	   need_init = .FALSE.
	   init_val = unspecified_val8
	   nbuffer = 1
	ENDIF

	RETURN

	END
