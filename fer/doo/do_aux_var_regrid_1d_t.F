	SUBROUTINE DO_AUX_VAR_REGRID_1D_T( axis,
     .				           src,  msrc,  src_cx,
     .				           dst,  mdst,  dst_cx,
     .				           tsrc, mtsrc, tsrc_cx,
     .					   ldst_min, ldst_max ) 

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
* perform a 1D t-axis "orthoganalization" regrid guided by an
* auxiliary variable containing a 2D array of time steps
* allow for the usual Ferret promotion of point axes to match span axes
* where either the source data or the position field may have degenerate axes
* (typically the position array will be degenerate in Z and E) 

* This code copied/modified from do_aux_var_regrid_1d_z.F

* V695+ 3/15 *sh*
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	
* calling argument declarations:
	INTEGER	axis,
     .		msrc, mdst, mtsrc, src_cx, dst_cx, tsrc_cx,
     .		ldst_min, ldst_max
        REAL    src ( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .                m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .          dst ( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .                m5lot:m5hit,m5loe:m5hie,m5lof:m5hif ),
     .          tsrc( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .                m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )


* internal variable declarations:
        INTEGER CX_DIM_LEN,
     .          idim,
     .          i,   j,   k,   l,   m,   n,
     .          is,  js,  ks,  ls,  ms,  ns,
     .          si0, sj0, sk0, sl0, sm0, sn0,
     .          dsi, dsj, dsk, dsl, dsm, dsn,
     .		ip,   jp,  kp,       mp
        INTEGER nmin, nmax, dgrid, status,
     .		lsrc_min, lsrc_max, lsrc_start, lsrc_end, ldst_start, ldst_end
        REAL    tsrc_min, tsrc_max
        REAL*8  TM_WORLD, depsilon
* internally declared work arrays
	REAL*8	tdst(ldst_min:ldst_max)

* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
* "s" is the source data (src)
* "p" (tsrcition) is the time steps of the source data ==> the 2D array
* "d" indicates the delta increment
* "0" indicates the start (lo) index
        INTEGER srcinc(nferdims), srclo(nferdims)
        EQUIVALENCE
     .	    (srcinc(1),dsi), (srcinc(2),dsj), (srcinc(3),dsk),
     .	    (srcinc(4),dsl), (srcinc(5),dsm), (srcinc(6),dsn),
     .	    (srclo(1), si0), (srclo(2), sj0), (srclo(3), sk0),
     .	    (srclo(4), sl0), (srclo(5), sm0), (srclo(6), sn0)

*==========================

* initialize
* nicer names for key index limits
	dgrid   = cx_grid(dst_cx)
	nmin    = cx_lo_ss(dst_cx,f_dim) ! lowest forecast axis index
	nmax    = cx_hi_ss(dst_cx,f_dim) !  hiest forecast axis index
	lsrc_min = cx_lo_ss(src_cx,t_dim) ! l index min in source data
	lsrc_max = cx_hi_ss(src_cx,t_dim) ! l index max in source data

* auxiliary TIME field is 2D on T-F so set up static indices for other axes
	ip = cx_lo_ss(tsrc_cx,x_dim)
	jp = cx_lo_ss(tsrc_cx,y_dim)
	kp = cx_lo_ss(tsrc_cx,z_dim)
	mp = cx_lo_ss(tsrc_cx,e_dim)

* extract the time step values of the destination
	DO l = ldst_min, ldst_max
	   tdst(l) = TM_WORLD(l, dgrid, t_dim, box_middle)
	ENDDO

* compute the delta increment for each axis
* this allows "promotion" of degenerate axes
        DO idim = 1, nferdims
           IF ( CX_DIM_LEN(idim,src_cx) .EQ. 1 ) THEN
              srcinc(idim) = 0
           ELSE
              srcinc(idim) = 1
           ENDIF
	ENDDO

* compute the starting subscript for each axis of each component
* used below through equivalence relationship
* pre-decrement by 1 delta for looping ahead)
        DO idim = 1, nferdims
	   srclo(idim) = cx_lo_ss(src_cx,idim) - srcinc(idim)
	ENDDO

* the relationship between the time steps in the source data (the
* values found in the "position" data -- the 2D aux time field)
* and the same times in the destination field looks something
* like this over 3 forecast cycles

*   **     2d aux times    **	      ** orthoganalized T-F field **
*                       		                 	110
*                       		                 	109
*                       		        1010   1010	108
*                       		        1009   1009	107
*   18     >1008   1012   1016		 1008   1008   1008	106
*   17     >1007   1011   1015		 1007   1007		105
*   16     >1006  >1010   1014		 1006   1006		104
*   15     >1005  >1009   1013		 1005   1005		103
*   14      1004  >1008   1012		 1004   1004		102
*   13      1003  >1007   1011		 			101
*   12      1002  >1006  >1010		 			100
*   11      1001  >1005  >1009		 			 99
*   10      1000   1004  >1008		 			 98
*    |								  |
* l in src_cx						    l in dst_cx 

* ">" above indicate a requested time range of T=1005:1010

* the outer loop is over the frequency axis
	ns = sn0
	DO n = mr_lo_s6(mdst), mr_hi_s6(mdst)
	   ns = ns + dsn

* are there time steps to be copied at this forecast index
	   tsrc_min = tsrc(ip,jp,kp,lsrc_min,mp,n) ! min time at this F
	   tsrc_max = tsrc(ip,jp,kp,lsrc_max,mp,n) ! max time at this F
* tolerance in time matching (would be better if it was consistent for all F)
           depsilon = (2.0**(-42)) *
     .             ABS( (tsrc_max-tsrc_min) / (lsrc_max-lsrc_min+1.0) )
        IF ( ((tdst(ldst_min) - tsrc_max) .GT. depsilon)
     .  .OR. ((tsrc_min - tdst(ldst_max)) .GT. depsilon) ) CYCLE  ! nothing to copy

* locate the index at which the block of times to be copied begins in source and dest
           IF ( (tsrc_min - tdst(ldst_min)) .GT. depsilon ) THEN
	      lsrc_start = lsrc_min
	      DO ldst_start = ldst_min, ldst_max
                  IF ( ABS(tdst(ldst_start) - tsrc_min) .LT. depsilon ) EXIT
	      ENDDO
	   ELSE
	      ldst_start = ldst_min
	      DO lsrc_start = lsrc_min, lsrc_max
                 IF ( ABS(tsrc(ip,jp,kp,lsrc_start,mp,n) -
     .                    tdst(ldst_min)) .LT. depsilon ) EXIT
	      ENDDO
	   ENDIF

* locate the index at which the block of times to be copied ends
	   IF ( (tdst(ldst_max) - tsrc_max) .GT. depsilon ) THEN
	      lsrc_end = lsrc_max
	   ELSE
	      DO lsrc_end = lsrc_max, lsrc_min, -1
                 IF ( ABS(tsrc(ip,jp,kp,lsrc_end,mp,n) -
     .                    tdst(ldst_max)) .LT. depsilon ) EXIT
	      ENDDO
	   ENDIF
	   ldst_end = ldst_start + (lsrc_end-lsrc_start)

* copy the src data to the destination array -- loop over its  X,Y,Z,T,E range
* the 's' variables are source array indices
 100	   is = si0
	   DO i = mr_lo_s1(mdst), mr_hi_s1(mdst)
	    is = is + dsi

	    js = sj0
	    DO j = mr_lo_s2(mdst), mr_hi_s2(mdst)
	     js = js + dsj

	     ks = sk0
	     DO k = mr_lo_s3(mdst), mr_hi_s3(mdst)
	      ks = ks + dsk

* ... special set-up for L axis
	      ls = lsrc_start - 1
	      DO l = ldst_start, ldst_end
	       ls = ls + 1

	       ms = sm0
	       DO m = mr_lo_s5(mdst), mr_hi_s5(mdst)
	        ms = ms + dsm

	        dst(i,j,k,l,m,n) = src(is,js,ks,ls,ms,ns)

	       ENDDO
	      ENDDO
	     ENDDO
	    ENDDO
	   ENDDO

	ENDDO  ! n loop

* successful completion.  Clean up.
        CALL RELEASE_DYN_WORK_SPACE   ! ?? needed ??
        RETURN

* error exit
 5000	RETURN
	END
