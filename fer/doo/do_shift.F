	INTEGER FUNCTION DO_SHIFT (	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					wkbuf	)

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
* by amount ishift

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/17/87
* revision 0.1 - 9/9/87  - changed from subroutine to function
* revision 0.2 - 10/4/88 - kludge fix for 1-D variables where dim1=dim2
* V200:   6/6/89 - 4D symmetrical - major re-write
*       10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V230:   8/3/92 - bug: if the requested region had unspecified limits then
*                  the component is full-span so subtracting shf results in
*                  exceeding subscript bounds

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, wkbuf(*), 
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER ii, shf, i, j, k, l, lo_sub(4), hi_sub(4)
	REAL	bad_res

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	shf = INT(arg)
	bad_res = mr_bad_data( res_mr )

* compute limits within which calculation is possible
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE
	lo_sub(idim) = cx_lo_ss(com_cx,idim) - shf
	hi_sub(idim) = cx_hi_ss(com_cx,idim) - shf
        IF ( lo_sub(idim) .LT. cx_lo_ss(res_cx,idim) ) 
     .                  lo_sub(idim) = cx_lo_ss(res_cx,idim)  ! 8/92
        IF ( hi_sub(idim) .GT. cx_hi_ss(res_cx,idim) ) 
     .                  hi_sub(idim) = cx_hi_ss(res_cx,idim)  ! 8/92

* SHIFT ALONG X
	IF ( idim .EQ. x_dim ) THEN
	   DO 100 l = lo_s4, hi_s4
	   DO 100 k = lo_s3, hi_s3
	   DO 100 j = lo_s2, hi_s2
	   DO 100 i = lo_s1, hi_s1
 100	   res(i,j,k,l) = com(i+shf,j,k,l)

* SHIFT ALONG Y
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 200 l = lo_s4, hi_s4
	   DO 200 k = lo_s3, hi_s3
	   DO 200 j = lo_s2, hi_s2
	   DO 200 i = lo_s1, hi_s1
 200	   res(i,j,k,l) = com(i,j+shf,k,l)

* SHIFT ALONG Z
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 300 l = lo_s4, hi_s4
	   DO 300 k = lo_s3, hi_s3
	   DO 300 j = lo_s2, hi_s2
	   DO 300 i = lo_s1, hi_s1
 300	   res(i,j,k,l) = com(i,j,k+shf,l)

* SHIFT ALONG T
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 400 l = lo_s4, hi_s4
	   DO 400 k = lo_s3, hi_s3
	   DO 400 j = lo_s2, hi_s2
	   DO 400 i = lo_s1, hi_s1
 400	   res(i,j,k,l) = com(i,j,k,l+shf)

	ENDIF

* always successful
	DO_SHIFT = ferr_ok

	RETURN
	END
