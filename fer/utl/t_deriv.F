	SUBROUTINE T_DERIV( com, mcom, res, mres, dt, reg )

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
* compute centered derivative of component along axis t_dim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/1/89 - based on DO_DERIV_CNTR
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* 2/12 *acm* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	LOGICAL reg
	INTEGER	mcom, mres
	REAL	   dt,
     .   com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .   res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	INTEGER i, j, k, l, m, n, grid, lo_l, hi_l, lm1, lp1
	REAL	bad_com, bad_res, lo_com, hi_com, box, box1
	REAL*8	TM_WORLD

* initialize
	grid    = mr_grid( mres )
	bad_com = mr_bad_data( mres )
	bad_res = mr_bad_data( mcom )

* fill in edges if not computable
	lo_l = mr_lo_s4(mres)
	IF ( mr_lo_s4(mcom) .EQ. mr_lo_s4(mres) ) THEN
	   DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	   DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	   DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      res(i,j,k,lo_l,m,n) = bad_res
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   lo_l = lo_l + 1
	ENDIF
	hi_l = mr_hi_s4(mres)
	IF ( mr_hi_s4(mcom) .EQ. mr_hi_s4(mres) ) THEN
	   DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	   DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	   DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      res(i,j,k,hi_l,m,n) = bad_res
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   hi_l = hi_l - 1
	ENDIF
	   
* loop through all computable points
	DO l = lo_l, hi_l

	lm1 = l-1
	lp1 = l+1

* time box size
	   box = TM_WORLD( lp1, grid, t_dim, box_middle )
     .	       - TM_WORLD( lm1, grid, t_dim, box_middle )

* all T spacings the same ?
	   IF ( l .EQ. lo_l ) THEN
	      box1 = box
	      reg  = .TRUE.
	   ELSE
	      IF ( box .NE. box1 ) reg = .FALSE.
	   ENDIF

	   DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	   DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	      lo_com = com(i,j,k,lm1,m,n)
	      hi_com = com(i,j,k,lp1,m,n)
	      IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	         res(i,j,k,l,m,n) = bad_res
	      ELSE
	         res(i,j,k,l,m,n) = (hi_com-lo_com) / box
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDDO

* compute delta t
	IF ( reg ) THEN
	   dt = box
	ELSE
	   box = TM_WORLD( hi_l+1, grid, t_dim, box_middle )
     .	       - TM_WORLD( lo_l-1, grid, t_dim, box_middle )
	   dt = box / (hi_l-lo_l+1)
	ENDIF

	RETURN
	END	
