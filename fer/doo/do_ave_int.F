	SUBROUTINE DO_AVE_INT(	action, idim, com_lo, com_hi,
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
* perform both averaging and integrating (definite) - since they differ only
* by a multiplying factor

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: ????
*      10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V230:  5/6/92 - COS correction checks that J location is defined
*       8/10/92 - weights were wrong for integration within a single box
*                 added "IF (point)" logic

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, need_cos, integrate, point
	INTEGER i, j, k, l, grid, pt, lo, hi, axis
	REAL	BOX_SIZE, unit, bad_com, bad_res, cos_factor
	REAL*8	TM_WORLD, dsum, bsum

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	INTEGER	action, idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		box( com_lo:com_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	axis = grid_line( idim, grid )
	pt   = cx_lo_ss( res_cx, idim )
	lo   = com_lo
	hi   = com_hi
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )
	integrate = action .EQ. isact_integ_def
        point = .FALSE.

* get and save grid box sizes (weights) just once for efficiency
* (end boxes may be only partially weighted)
        IF ( lo .EQ. hi ) THEN    ! 8/92
           box(lo) = cx_hi_ww(idim,res_cx) - cx_lo_ww(idim,res_cx)
           point = box(lo) .EQ. 0.0
        ELSE
	   box(lo) = TM_WORLD(lo, grid, idim, box_hi_lim)
     .		   - cx_lo_ww(idim,res_cx)
	   DO 100 i = lo+1, hi-1
 100	   box(i)  = BOX_SIZE( i, grid, idim )
	   box(hi) = cx_hi_ww(idim,res_cx)
     .		   - TM_WORLD(hi, grid, idim, box_lo_lim)
        ENDIF

* perform integrations on standard units if possible
	IF ( integrate ) THEN
	   unit = un_convert( line_unit_code(axis) )
	   DO 110 i = lo, hi
 110	   box(i) = box(i) * unit
	ENDIF

* AVERAGE/INTEGRATE
* (is double precision summing necessary ? *sh* 6/89)
* ALONG X AXIS
* (note: X integration is a special case 'cuz it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. cx_lo_s2(com_cx).NE.unspecified_int4
	   DO 210 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   IF ( need_cos ) cos_factor = COS( deg2rad *       
     .             SNGL( TM_WORLD( j, grid, y_dim, box_middle )) ) 
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
              IF ( point ) THEN  ! 8/92
                 IF ( com(lo,j,k,l) .EQ. bad_com ) THEN
                    res(pt,j,k,l) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(pt,j,k,l) = 0.0
                 ELSE
                    res(pt,j,k,l) = com(lo,j,k,l)
                 ENDIF
                 GOTO 210
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 200 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 200
	         bsum = bsum + box(i)
	         dsum = dsum + box(i)*com(i,j,k,l)
 200	      CONTINUE
	      IF ( bsum .EQ. 0.0D0 ) THEN
	         res(pt,j,k,l) = bad_res
	      ELSEIF (integrate) THEN
	         IF ( need_cos ) THEN
	            res(pt,j,k,l) = dsum * cos_factor
	         ELSE
	            res(pt,j,k,l) = dsum
	         ENDIF	
	      ELSE
	         res(pt,j,k,l) = dsum/bsum
	      ENDIF
 210	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN   ! 8/92
                 IF ( com(i,lo,k,l) .EQ. bad_com ) THEN
                    res(i,pt,k,l) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,pt,k,l) = 0.0
                 ELSE
                    res(i,pt,k,l) = com(i,lo,k,l)
                 ENDIF
                 GOTO 310
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 300 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 300
	         bsum = bsum + box(j)
	         dsum = dsum + box(j)*com(i,j,k,l)
 300	      CONTINUE
	      IF ( bsum .EQ. 0.0D0 ) THEN
	         res(i,pt,k,l) = bad_res
	      ELSEIF (integrate) THEN
	         res(i,pt,k,l) = dsum
	      ELSE
	         res(i,pt,k,l) = dsum/bsum
	      ENDIF
 310	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 410 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN    ! 8/92
                 IF ( com(i,j,lo,l) .EQ. bad_com ) THEN
                    res(i,j,pt,l) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,pt,l) = 0.0
                 ELSE
                    res(i,j,pt,l) = com(i,j,lo,l)
                 ENDIF
                 GOTO 410
	      ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 400 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 400
	         bsum = bsum + box(k)
	         dsum = dsum + box(k)*com(i,j,k,l)
 400	      CONTINUE
              IF ( bsum .EQ. 0.0D0 ) THEN
	      res(i,j,pt,l) = bad_res
	      ELSEIF (integrate) THEN
	         res(i,j,pt,l) = dsum
	      ELSE
	         res(i,j,pt,l) = dsum/bsum
	      ENDIF
 410	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 510 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 510 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 510 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
                 IF ( com(i,j,k,lo) .EQ. bad_com ) THEN
                    res(i,j,k,pt) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,k,pt) = 0.0
                 ELSE
                    res(i,j,k,pt) = com(i,j,k,lo)
                 ENDIF
                 GOTO 510
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 500 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 500
	         bsum = bsum + box(l)
	         dsum = dsum + box(l)*com(i,j,k,l)
 500	      CONTINUE
	      IF ( bsum .EQ. 0.0D0 ) THEN
	         res(i,j,k,pt) = bad_res
	      ELSEIF (integrate) THEN
	         res(i,j,k,pt) = dsum
	      ELSE
	         res(i,j,k,pt) = dsum/bsum
	      ENDIF
 510	   CONTINUE
	ENDIF

* success
	RETURN

* error exit
! 5000	RETURN
	END	
