	SUBROUTINE DO_VAR_SUB(	idim, com_lo, com_hi,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				box )

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
* (what about COSINE weighting along Y axis if in XY plane ??)
* determine the variance along the indicated axis
* nested with DO_VARIANCE in order to ease the dimension checking of box

* calculation is done in two passes to accomodate unequal weighting and
* to increase numerical stability 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:   3/2/92
* V420: 11/95 - converted diff and mean to DP so variance is calc in DP
*	 2/96 - fix bug introduced in 11/95: CGRID_AXIS has wrong args

* internal variable declarations:
!	LOGICAL	GEOG_COS_FACTOR, need_cos
	INTEGER  CGRID_AXIS, i, j, k, l, grid, pt, lo, hi, axis, npt
	REAL	BOX_SIZE, bad_com, bad_res
!       REAL    cos_factor
	REAL*8	TM_WORLD, dsum, bsum, mean, diff

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		box( com_lo:com_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	axis = CGRID_AXIS( idim, com_cx )
	pt   = cx_lo_ss( res_cx, idim )
	lo   = com_lo
	hi   = com_hi
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* get and save grid box sizes (weights) just once for efficiency
* (end boxes may be only partially weighted)
	box(lo) = TM_WORLD(lo, grid, idim, box_hi_lim)
     .		- cx_lo_ww(idim,res_cx)
	DO 100 i = lo+1, hi-1
 100	box(i)  = BOX_SIZE( i, grid, idim )
	box(hi) = cx_hi_ww(idim,res_cx)
     .		- TM_WORLD(hi, grid, idim, box_lo_lim)

* ALONG X AXIS
* (note: X integration is a special case 'cuz it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   DO 210 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
* ... Pass #1: determine mean
              npt = 0
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 200 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 200
	         bsum = bsum + box(i)
	         dsum = dsum + box(i)*com(i,j,k,l)
                 npt = npt + 1
 200	      CONTINUE
	      IF ( npt .LE. 1 ) THEN
	         res(pt,j,k,l) = bad_res
                 GOTO 210
	      ELSE
                 mean = dsum / bsum
              ENDIF
* ... Pass #2: determine variance
              dsum = 0.0D0    ! sum of squares
              DO 205 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
                 IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 205
                 diff = com(i,j,k,l) - mean
                 dsum = dsum + box(i)*diff*diff
 205          CONTINUE
              res(pt,j,k,l) = dsum / bsum
 210	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
!	   need_cos = GEOG_COS_FACTOR( idim, grid )
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
!	      IF ( need_cos ) cos_factor = COS( deg2rad *       
!     .             SNGL( TM_WORLD( j, grid, y_dim, box_middle )) ) 
* ... Pass #1: determine mean
              npt = 0
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 300 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 300
	         bsum = bsum + box(j)
	         dsum = dsum + box(j)*com(i,j,k,l)
                 npt = npt + 1
 300	      CONTINUE
	      IF ( npt .LE. 1 ) THEN
	         res(i,pt,k,l) = bad_res
                 GOTO 310
	      ELSE
                 mean = dsum / bsum
              ENDIF
* ... Pass #2: determine variance
              dsum = 0.0D0    ! sum of squares
	      DO 305 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
                 IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 305
                 diff = com(i,j,k,l) - mean
                 dsum = dsum + box(j)*diff*diff
 305          CONTINUE
              res(i,pt,k,l) = dsum / bsum
 310	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 410 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
* ... Pass #1: determine mean
              npt = 0
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 400 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 400
	         bsum = bsum + box(k)
	         dsum = dsum + box(k)*com(i,j,k,l)
                 npt = npt + 1
 400	      CONTINUE
	      IF ( npt .LE. 1 ) THEN
	         res(i,j,pt,l) = bad_res
                 GOTO 410
	      ELSE
                 mean = dsum / bsum
              ENDIF
* ... Pass #2: determine variance
              dsum = 0.0D0    ! sum of squares
	      DO 405 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
                 IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 405
                 diff = com(i,j,k,l) - mean
                 dsum = dsum + box(k)*diff*diff
 405          CONTINUE
              res(i,j,pt,l) = dsum / bsum
 410	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 510 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 510 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 510 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
* ... Pass #1: determine mean
              npt = 0
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 500 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 500
	         bsum = bsum + box(l)
	         dsum = dsum + box(l)*com(i,j,k,l)
                 npt = npt + 1
 500	      CONTINUE
	      IF ( npt .LE. 1 ) THEN
	         res(i,j,k,pt) = bad_res
                 GOTO 510
	      ELSE
                 mean = dsum / bsum
              ENDIF
* ... Pass #2: determine variance
              dsum = 0.0D0    ! sum of squares
	      DO 505 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
                 IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 505
                 diff = com(i,j,k,l) - mean
                 dsum = dsum + box(l)*diff*diff
 505          CONTINUE
              res(i,j,k,pt) = dsum / bsum
 510	   CONTINUE
	ENDIF

* success
	RETURN

* error exit
! 5000	RETURN
	END	
