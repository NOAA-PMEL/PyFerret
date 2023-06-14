	INTEGER FUNCTION DO_SHIFT (	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					wkbuf	)

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
* shift the data from a component grid into the result grid along axis idim
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
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xprog_state.cmn'
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, wkbuf(*), 
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	LOGICAL TM_ITSA_DSG
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, DO_SHIFT_DSG,
     .          ii, shf, i, j, k, l, m, n,
     .          lo_sub(nferdims), hi_sub(nferdims),
     .          grid, dset, nfeatures, mr_list(2)
	REAL	bad_res

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6,
     .          hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (lo_s5, lo_sub(5)) , (lo_s6, lo_sub(6)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4)),
     .		    (hi_s5, hi_sub(5)) , (hi_s6, hi_sub(6))

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )



	grid = mr_grid( com_mr )

	IF (TM_ITSA_DSG(grid)) THEN 
* DSG ragged array
	
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)

	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, dsg_orientation(dset) )

	   DO_SHIFT = DO_SHIFT_DSG (	idim, arg, dset, nfeatures,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					wkbuf	)
	   GOTO 5000

	ENDIF  ! not dsg

* initialize
	shf = INT(arg)
	bad_res = mr_bad_data( res_mr )

* compute limits within which calculation is possible
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO
	lo_sub(idim) = cx_lo_ss(com_cx,idim) - shf
	hi_sub(idim) = cx_hi_ss(com_cx,idim) - shf
        IF ( lo_sub(idim) .LT. cx_lo_ss(res_cx,idim) ) 
     .                  lo_sub(idim) = cx_lo_ss(res_cx,idim)  ! 8/92
        IF ( hi_sub(idim) .GT. cx_hi_ss(res_cx,idim) ) 
     .                  hi_sub(idim) = cx_hi_ss(res_cx,idim)  ! 8/92

* SHIFT ALONG X
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i+shf,j,k,l,m,n)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SHIFT ALONG Y
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i,j+shf,k,l,m,n)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SHIFT ALONG Z
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i,j,k+shf,l,m,n)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SHIFT ALONG T
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i,j,k,l+shf,m,n)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SHIFT ALONG E
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i,j,k,l,m+shf,n)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SHIFT ALONG F
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      res(i,j,k,l,m,n) = com(i,j,k,l,m,n+shf)
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

* always successful
 5000	DO_SHIFT = ferr_ok

	RETURN
	END
