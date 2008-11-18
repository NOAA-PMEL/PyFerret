	SUBROUTINE DO_4D_STRING_GOODBAD( trans,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx, arg )

*
*
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
* perform a multi-axis transformation

* multi-axis transformations are those special cases where sequential
* application of single-axis transforms may yield improper results. For
* example, averaging within a circular region would give inappropriate
* results if carried out sequentially: the first axis of averaging would
* be correct but the second axis would weight each averaged row from the
* first axis equally, despite the fact that they represented different
* numbers of data points.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 11/2008 - based on DO_4d_string_goodbad
* 
* Compute NGD NBD for string variables.


* internal variable declarations:
	LOGICAL point, yes_ax(4), no_ax(4),
     .		at_ave, at_var, at_din, at_nbd, at_ngd
	INTEGER idim, i, j, ir, jr, kr, lr, ic, jc, kc, lc, nbd, ngd,
     .		com_lo(4), com_hi(4), ind_lo, ind_hi
	REAL	bad_com, bad_res

	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'tmap_dims.parm'


* calling argument declarations:
	INTEGER	trans, com_mr, com_cx, res_mr, res_cx
	REAL*8  com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )
	REAL	res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* equivalences
	LOGICAL  yes_xax, yes_yax, yes_zax, yes_tax,
     .		 no_xax,  no_yax,  no_zax,  no_tax

	INTEGER GET_C_STRING_LEN, slen, 
     .		i_lo, j_lo, k_lo, l_lo,
     .		i_hi, j_hi, k_hi, l_hi

	EQUIVALENCE ( yes_ax(1), yes_xax ),
     .		    ( yes_ax(2), yes_yax ),
     .		    ( yes_ax(3), yes_zax ),
     .		    ( yes_ax(4), yes_tax ),
     .		    (  no_ax(1),  no_xax ),
     .		    (  no_ax(2),  no_yax ),
     .		    (  no_ax(3),  no_zax ),
     .		    (  no_ax(4),  no_tax ),
     .		    (  com_lo(1),  i_lo ),
     .		    (  com_lo(2),  j_lo ),
     .		    (  com_lo(3),  k_lo ),
     .		    (  com_lo(4),  l_lo ),
     .		    (  com_hi(1),  i_hi ),
     .		    (  com_hi(2),  j_hi ),
     .		    (  com_hi(3),  k_hi ),
     .		    (  com_hi(4),  l_hi )

* --- end of introductory code ---

* initialize
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )
        point = .FALSE.

	at_nbd = trans .EQ. trans_4d_bad_pt
	at_ngd = trans .EQ. trans_4d_good_pt

* determine the axes to be transforomed
	DO 10 idim = 1, 4
	  yes_ax(idim) = cx_trans(idim,res_cx) .EQ. trans
	  no_ax (idim) = .NOT.yes_ax(idim)
 10	CONTINUE

* do the transformation:  ngood, nbad only
* Write a note about missing values of strings 

	IF (at_ngd) THEN
	   CALL WARN
     .    ('@NGD of string variable, counting null strings as missing')
	ELSEIF (at_nbd) THEN
	   CALL WARN
     .    ('@NBD of string variable, counting null strings as missing')
	ENDIF

* Note: the looping structure allows a possible mix of axis transforms
*	to be performed in a single framework
* ... set up default limits (com_lo, com_hi) in preparation for the inner loop
* "r" refers to result, "c" refers to component

* Set the range of loop indices for the component data on the transformed axes
	DO 150 idim = 1, 4
	  IF (yes_ax(idim)) THEN
	    com_lo(idim) = cx_lo_ss(com_cx,idim)
	    com_hi(idim) = cx_hi_ss(com_cx,idim)
	  ENDIF
 150	CONTINUE


* LOOP OVER THE FULL RANGE OF THE RESULT
* ... also set up loop indices for non-transformed axes
*     (component loop to include only the single result point on "no" axes)
	DO 300 lr = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	IF (no_tax) THEN
	   l_lo = lr
	   l_hi = lr
	ENDIF
	DO 300 kr = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	IF (no_zax) THEN
	  k_lo = kr
	  k_hi = kr
	ENDIF
	DO 300 jr = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	IF (no_yax) THEN
	  j_lo = jr
	  j_hi = jr
	ENDIF
	DO 300 ir = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
	IF (no_xax) THEN
	  i_lo = ir
	  i_hi = ir
	ENDIF

* pathological case: world coordinate range of zero along an xform axis
          IF ( point ) THEN    ! 8/92
	    IF (at_din) THEN
               IF (com(ic,jc,kc,lc) .EQ. bad_com) THEN
                  res(ir,jr,kr,lr) = bad_res
               ELSE
                  res(ir,jr,kr,lr) = 0.0
	       ENDIF
            ELSE
               res(ir,jr,kr,lr) = bad_res
            ENDIF
            GOTO 300
          ENDIF

* initialize the inner loop accumulators

	  ngd  = 0
	  nbd  = 0

* now loop over the component data accumulating results along relevant axes

	  DO 210 lc = l_lo, l_hi
	  DO 210 kc = k_lo, k_hi
	  DO 210 jc = j_lo, j_hi
	  DO 210 ic = i_lo, i_hi
             slen = GET_C_STRING_LEN(com(ic,jc,kc,lc))
             IF ( slen .GT. 0 ) THEN
                ngd = ngd + 1
             ELSE
                nbd = nbd + 1
             ENDIF
 210	  CONTINUE

* compute result from accumulators
	  IF (at_ngd) THEN
	     res(ir,jr,kr,lr) = ngd
	  ELSEIF (at_nbd) THEN
	     res(ir,jr,kr,lr) = nbd
	  ENDIF

 300	CONTINUE

	RETURN
 
	END	

