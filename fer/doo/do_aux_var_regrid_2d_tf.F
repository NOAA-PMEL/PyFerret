	SUBROUTINE DO_AUX_VAR_REGRID_2D_TF(
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
* V698  3/16 *sh* - compute f_delta from the time2d field rather than F axis
*                   allowing the F axis to be redefined
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xtm_grid.cmn_text'   ! for line_delta
	
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
        INTEGER CGRID_AXIS,
     .		ndst_min, ndst_max, dgrid,
     .		lsrc_min, lsrc_max, lsrc_start, lsrc_end, ldst_start, ldst_end,
     .		nsrc_min, nsrc_max, nsrc0, lsrc0, f_axis, l_per_n, ldst0
        REAL    tsrc_min, tsrc_max, t_target, f_delta, t_delta
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
	ndst_min    = cx_lo_ss(dst_cx,f_dim)  ! lowest forecast axis index
	ndst_max    = cx_hi_ss(dst_cx,f_dim)  !  hiest forecast axis index
	lsrc_min = cx_lo_ss(src_cx,t_dim) ! l index min in source data
	lsrc_max = cx_hi_ss(src_cx,t_dim) ! l index max in source data
	nsrc_min = cx_lo_ss(src_cx,f_dim)
	nsrc_max = cx_hi_ss(src_cx,f_dim)

* auxiliary TIME field is 2D on T-F so set up static indices for other axes
	ip = cx_lo_ss(tsrc_cx,x_dim)
	jp = cx_lo_ss(tsrc_cx,y_dim)
	kp = cx_lo_ss(tsrc_cx,z_dim)
	mp = cx_lo_ss(tsrc_cx,e_dim)

* f_delta is the time lag between forecasts.
* It should be consistent in tsrc and should match the delta of the F lag axis
! 3/16 -- do not get it from the F axis, since the user may alter that
!	f_axis = CGRID_AXIS(f_dim, dst_cx)
!	f_delta = line_delta(f_axis)
	f_delta = (tsrc(ip,jp,kp,m1lot,mp,m1hif)
     .	         - tsrc(ip,jp,kp,m1lot,mp,m1lof))
     .		                   / (m1hif-m1lof+1)

* max insignificant time difference
        depsilon = (2.0**(-42)) * ABS(f_delta)

* extract the time step values of the destination
	DO l = ldst_min, ldst_max
	   tdst(l) = TM_WORLD(l, dgrid, t_dim, box_middle)
	ENDDO

* no source data overlapping with requested dest times?  Then quit here
       IF ( (tdst(ldst_min) - tsrc(ip,jp,kp,lsrc_max,mp,nsrc_max)) .GT. depsilon
     . .OR. (tsrc(ip,jp,kp,lsrc_min,mp,nsrc_min) - tdst(ldst_max)) .GT. depsilon )
     .								RETURN
* number of time steps before the start of the next forecast
	t_delta = (tsrc(ip,jp,kp,m1hit,mp,m1lof)
     .	         - tsrc(ip,jp,kp,m1lot,mp,m1lof))
     .		                   / (m1hit-m1lot+1)
* ... for irregular monthly time steps, t_delta is approximate
	l_per_n = INT(f_delta/t_delta + 0.5)

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

* the relationship between the time steps in the 2D aux time field (cx1)
* and the same times in the destination field looks something
* like this

*          **  2d aux times    **  ==>   ** cal-T x lag-F field **

*   12  1011(0)   1015(1)   1019(2)        1019(2) 1019(3) 1019(4)     20
*   11  1010(0)   1014(1)   1018(2)        1018(2) 1018(3) 1018(4)     19
*   10  1009(0)   1013(1)   1017(2)        1017(2) 1017(3) 1017(4)     18
*    9  1008(0)   1012(1)   1016(2)        1016(2) 1016(3) 1016(4)     17
*    8  1007(0)   1011(1)   1015(2)        1015(1) 1015(2) 1015(3)     16
*    7  1006(0)  >1010(1)   1014(2)        1014(1) 1014(2) 1014(3)     15
*    6  1005(0)   1009(1)   1013(2)        1013(1) 1013(2) 1013(3)     14
*    5  1004(0)   1008(1)   1012(2)        1012(1) 1012(2) 1012(3)     13
*    4  1003(0)   1007(1)   1011(2)        1011(0) 1011(1) 1011(2)     12
*    3  1002(0)   1006(1)   1010(2)        1010(0) 1010(1) 1010(2)     11
*    2  1001(0)  >1005(1)   1009(2)        1009(0) 1009(1) 1009(2)     10
*    1  1000(0)   1004(1)   1008(2)        1008(0) 1008(1) 1008(2)      9
*    ^                                                                  ^
*    |                                                                  |
* l in src_cx                                                     l in dst_cx 

* ">" above indicate a requested time range of T=1005:1010

* strategy for reorganizing the data:
*   - initialize by finding the last forecast (highest N index) which
*     contains the starting destination time requested.  Call this nsrc0
*   - loop over the destination time axis.
*     count when F-delta advances occur.  When they occur, advance to  
*     next nsrc0
*   - at each time, loop over the F axis,
*        start by copying nsrc0 source into n=1
*        advance to n=2,3, ... keeping T constant by incrementing L forward
*        at each consecutive forecast by an amount delta-F

* find the highest source N index at which lowest destination time can be found
	DO ldst0 = ldst_min, ldst_max
	   t_target = tdst(ldst0)
	   DO nsrc0 = cx_hi_ss(tsrc_cx,f_dim),cx_lo_ss(tsrc_cx,f_dim), -1
              IF ( (tsrc(ip,jp,kp,lsrc_min,mp,nsrc0) - t_target) .LE. depsilon ) THEN
	         DO lsrc0 = lsrc_min,lsrc_max
                    IF ( ABS(tsrc(ip,jp,kp,lsrc0,mp,nsrc0) - t_target) .LE. depsilon ) THEN
	               GOTO 50
	            ELSEIF  (tsrc(ip,jp,kp,lsrc0,mp,nsrc0) .GT. t_target) THEN
	               CALL WARN('1D time axis of FMRC contains time'//
     .			         ' not in forecast collection'//pCR//
     .				 'Results may not be correct.')
	               GOTO 50
	            ENDIF
	         ENDDO
	      ENDIF
	   ENDDO
* ... the time t_target does not exist in this forcast
	ENDDO
* can never get here: no overlap between source and destination time ranges
	STOP 'do_aux_ver_regrid_2d_tf'

* determine where we are starting on the l_per_n cycle
 50	lsrc0 = lsrc0 - 1

******************************************************************
	DO l = ldst0, ldst_max         ! ***** T-cal loop  ******

* advance to the next forecast for pulling from the source data
* (do not do this on initial entry -- everything set properly initially)
	 IF (l.GT.ldst0 .AND. MOD(ls,l_per_n).EQ.0) THEN
	   nsrc0 = nsrc0 + 1
	   lsrc0 = lsrc_min - 1
	 ENDIF

	 lsrc0 = lsrc0 + 1
	 ls = lsrc0
	 ns    = nsrc0
	 n     = 1

	 DO WHILE (n<=ndst_max)          ! ***** N-lag loop *****

* this IF allows i) user N limits to exclude low n values
*           and ii) copying the ragged high end of the FMRC block
	   IF (n  .GE. ndst_min
     .   .AND. ns .LE. nsrc_max) THEN

* copy the src data to the destination array -- loop over its  X,Y,Z,E range
* the 's' variables are source array indices
	     is = si0
	     DO i = mr_lo_s1(mdst), mr_hi_s1(mdst)
	      is = is + dsi

	      js = sj0
	      DO j = mr_lo_s2(mdst), mr_hi_s2(mdst)
	       js = js + dsj

	       ks = sk0
	       DO k = mr_lo_s3(mdst), mr_hi_s3(mdst)
	        ks = ks + dsk

	         ms = sm0
	         DO m = mr_lo_s5(mdst), mr_hi_s5(mdst)
	          ms = ms + dsm

	          dst(i,j,k,l,m,n) = src(is,js,ks,ls,ms,ns)

	         ENDDO
	       ENDDO
	      ENDDO
	     ENDDO
	     
	   ENDIF

* set up for same destination time taken from the previous forecast
	   n = n + 1
	   ns = ns - 1
	   ls = ls + l_per_n 
	   IF (ns.LT.nsrc_min .OR. ls.GT.lsrc_max) EXIT
	 ENDDO     ! N loop

	ENDDO     ! L loop

* successful completion.  Clean up.
        CALL RELEASE_DYN_WORK_SPACE    ! ?? needed ??
        RETURN

* error exit
 5000	RETURN
	END
