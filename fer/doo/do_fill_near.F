	INTEGER FUNCTION DO_FILL_NEAR(  idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					buff	)

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
* fill along axis idim by duplicating the closest available valid pt

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  3/23/92 - based on DO_FILL_AVE
*         7/2/92 - changed logic so if 2 equi-distant points use the mean
*                - corrected subscriptong bug in L axis search

* Note:  this routine could be made simpler and faster by arranginr single
*        inner llops that looped "loop=1,edge" looking at index-loop and
*        index+loop to find the first valid point(s)

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER i, j, k, l, ii, jj, kk, ll, lo_sub(4), hi_sub(4),
     .		ii_com, lo_edge, hi_edge, lo_pos, hi_pos, edge
	REAL	bad_res, lo_coord, hi_coord, coord, lo_val, hi_val

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* local parameter declarations
	INTEGER     no_point
	PARAMETER ( no_point = -1 )

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	bad_res = mr_bad_data( com_mr )
	edge = arg

* compute limits within which calculation is straightforward
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE
	lo_edge = MAX( lo_sub(idim)-edge, cx_lo_ss(com_cx, idim) )
	hi_edge = MIN( hi_sub(idim)+edge, cx_hi_ss(com_cx, idim) )

* FILL ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
           lo_pos = no_point
           hi_pos = no_point
           DO 101 i = lo_edge, lo_s1
              IF ( com(i,j,k,l) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l)
                 lo_pos = i
              ENDIF
 101       CONTINUE
	   DO 110 i = lo_s1, hi_s1
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
                 lo_pos = i
	         GOTO 110
	      ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. i ) THEN
                 DO 104 ii = i+1, hi_edge
                    IF ( com(ii,j,k,l) .NE. bad_res ) THEN
                       hi_pos = ii
                       hi_val = com(ii,j,k,l)
                       GOTO 108
                    ENDIF
 104             CONTINUE
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 108          IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = bad_res
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = lo_val
                 ELSEIF ( i-lo_pos .EQ. hi_pos-i ) THEN
                    res(i,j,k,l) = ( lo_val + hi_val ) /2.
                 ELSEIF ( i-lo_pos .LT. hi_pos-i ) THEN
                    res(i,j,k,l) = lo_val
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ENDIF
 110	   CONTINUE

* FILL ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO 201 j = lo_edge, lo_s2
              IF ( com(i,j,k,l) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l)
                 lo_pos = j
              ENDIF
 201       CONTINUE
	   DO 210 j = lo_s2, hi_s2
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
                 lo_pos = j
	         GOTO 210
	      ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. j ) THEN
                 DO 204 jj = j+1, hi_edge
                    IF ( com(i,jj,k,l) .NE. bad_res ) THEN
                       hi_pos = jj
                       hi_val = com(i,jj,k,l)
                       GOTO 208
                    ENDIF
 204             CONTINUE
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 208          IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = bad_res
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = lo_val
                 ELSEIF ( j-lo_pos .EQ. hi_pos-j ) THEN
                    res(i,j,k,l) = ( lo_val + hi_val ) /2.
                 ELSEIF ( j-lo_pos .LT. hi_pos-j ) THEN
                    res(i,j,k,l) = lo_val
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ENDIF
 210	   CONTINUE

* FILL ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO 301 k = lo_edge, lo_s3
              IF ( com(i,j,k,l) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l)
                 lo_pos = k
              ENDIF
 301       CONTINUE
	   DO 310 k = lo_s3, hi_s3
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
                 lo_pos = k
	         GOTO 310
              ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. k ) THEN
                 DO 304 kk = k+1, hi_edge
                    IF ( com(i,j,kk,l) .NE. bad_res ) THEN
                       hi_pos = kk
                       hi_val = com(i,j,kk,l)
                       GOTO 308
                    ENDIF
 304             CONTINUE
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 308          IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = bad_res
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = lo_val
                 ELSEIF ( k-lo_pos .EQ. hi_pos-k ) THEN
                    res(i,j,k,l) = ( lo_val + hi_val ) /2.
                 ELSEIF ( k-lo_pos .LT. hi_pos-k ) THEN
                    res(i,j,k,l) = lo_val
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ENDIF
 310	   CONTINUE

* FILL ALONG T AXIS
	ELSE
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO 401 l = lo_edge, lo_s4
              IF ( com(i,j,k,l) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l)
                 lo_pos = l
              ENDIF
 401       CONTINUE
	   DO 410 l = lo_s4, hi_s4
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
                 lo_pos = l
	         GOTO 410
	      ENDIF
              IF ( com(i,j,k,l) .NE. bad_res ) THEN
                 res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
                 lo_pos = l
                 GOTO 410
              ENDIF
              IF ( hi_pos .LT. l ) THEN
* ... look forward for next valid point
                 DO 404 ll = l+1, hi_edge
                    IF ( com(i,j,k,ll) .NE. bad_res ) THEN
                       hi_pos = ll
                       hi_val = com(i,j,k,ll)
                       GOTO 408
                    ENDIF
 404             CONTINUE
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 408          IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = bad_res
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l) = lo_val
                 ELSEIF ( l-lo_pos .EQ. hi_pos-l ) THEN
                    res(i,j,k,l) = ( lo_val + hi_val ) /2.
                 ELSEIF ( l-lo_pos .LT. hi_pos-l ) THEN
                    res(i,j,k,l) = lo_val
                 ELSE
                    res(i,j,k,l) = hi_val
                 ENDIF
              ENDIF
 410	   CONTINUE
	ENDIF

* always success
	DO_FILL_NEAR = ferr_ok
	RETURN

	END	
