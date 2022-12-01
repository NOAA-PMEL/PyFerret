	SUBROUTINE DO_POINTS(	action, idim,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

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
* determine the number of good and number of bad points along the given axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:   2/3/92 - based on DO_AVE_INT
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V71 *acm* 12/16 this routine now performs accumulation, possibly continuing
*                 over several chunks, so keep adding to the count in res(). 
*                 Initialization and finalizing called separately from IS_TRANS
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 


        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* internal variable declarations:
	LOGICAL	good_un
	INTEGER i, j, k, l, m, n, pt, ngd, nbd
	REAL	bad_com, bad_res

* calling argument declarations:
	INTEGER	action, idim, com_mr, com_cx, res_mr, res_cx
	REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* --- end of introductory code ---

* initialize
        pt   = cx_lo_ss( res_cx, idim )
	bad_com = mr_bad_data( com_mr )
        bad_res = mr_bad_data( res_mr )
	good_un  = action .EQ. isact_good_pt

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
              ngd = 0
              nbd = 0
	      DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
	      ENDDO
              IF ( good_un ) THEN
                 res(pt,j,k,l,m,n) = res(pt,j,k,l,m,n) + ngd
              ELSE
                 res(pt,j,k,l,m,n) = res(pt,j,k,l,m,n) + nbd
              ENDIF
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
	      ENDDO
              IF ( good_un ) THEN
                 res(i,pt,k,l,m,n) = res(i,pt,k,l,m,n) + ngd
              ELSE
                 res(i,pt,k,l,m,n) = res(i,pt,k,l,m,n) + nbd
              ENDIF
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
	      ENDDO
              IF ( good_un ) THEN
                 res(i,j,pt,l,m,n) = res(i,j,pt,l,m,n) + ngd
              ELSE
                 res(i,j,pt,l,m,n) = res(i,j,pt,l,m,n) + nbd
              ENDIF
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG T AXIS
 	ELSEIF ( idim .EQ. t_dim ) THEN
 	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
 	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
 	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
 	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
 	      DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
 	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                     nbd = nbd + 1
                  ELSE
                     ngd = ngd + 1
                  ENDIF
 	      ENDDO
               IF ( good_un ) THEN
                  res(i,j,k,pt,m,n) = res(i,j,k,pt,m,n) + ngd
               ELSE
                  res(i,j,k,pt,m,n) = res(i,j,k,pt,m,n) + nbd
               ENDIF
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG E AXIS
 	ELSEIF ( idim .EQ. e_dim ) THEN
 	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
 	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
 	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
 	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
               ngd = 0
               nbd = 0
 	      DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
 	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                     nbd = nbd + 1
                  ELSE
                     ngd = ngd + 1
                  ENDIF
 	      ENDDO
               IF ( good_un ) THEN
                  res(i,j,k,l,pt,n) = res(i,j,k,l,pt,n) + ngd
               ELSE
                  res(i,j,k,l,pt,n) = res(i,j,k,l,pt,n) + nbd
               ENDIF
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 
* ALONG F AXIS
 	ELSEIF ( idim .EQ. f_dim ) THEN
 	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
 	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
 	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
 	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
               ngd = 0
               nbd = 0
 	      DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
 	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
                     nbd = nbd + 1
                  ELSE
                     ngd = ngd + 1
                  ENDIF
 	      ENDDO
               IF ( good_un ) THEN
                  res(i,j,k,l,m,pt) = res(i,j,k,l,m,pt) + ngd
               ELSE
                  res(i,j,k,l,m,pt) = res(i,j,k,l,m,pt) + nbd
               ENDIF
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

	ENDIF

	RETURN
	END
