	INTEGER FUNCTION DO_FILL_INTERP(idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					coords	)

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
* fill along axis idim by interpolating from the closest available valid pt

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  3/23/92 - based on DO_FILL_AVE
* V301:  2/15/94 - upgrade to accomodate modulo axes
* V402:   8/9/95 - bug fix: T=@FLN had "lo_coord = coords(k-pos0)"
* V550 *sh* 8/2/02 - support for modulo lengths. ensure that filling 
*		    does not operate acros the subspan modulo void point

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, coords(*),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	LOGICAL	TM_ITS_SUBSPAN_MODULO,
     .		modulo_axis, modulo, no_lo_val, has_mod_void
	INTEGER TM_MODULO_LINE_DIM, CGRID_AXIS,
     .		i, j, k, l, ii, jj, kk, ll, lo_sub(4), hi_sub(4),
     .		lo_edge, hi_edge, pos0, edge, lo_index, hi_index,
     .		lo_com, hi_com, line, axdim, look,
     .		coord_lo_ss, coord_hi_ss, next_hi_ss
	REAL	bad_res, lo_coord, hi_coord, coord, lo_val, hi_val,
     .		axlen
	REAL*8	TM_MODULO_AXLEN

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* local parameter declarations
	INTEGER     no_point
	PARAMETER ( no_point = -99999 )

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	bad_res = mr_bad_data( com_mr )
	edge = arg

* limits of the component data provided
	lo_com = cx_lo_ss(com_cx, idim)
	hi_com = cx_hi_ss(com_cx, idim)

* is it a modulo axis with sufficient component span provided to wrap around
	line = CGRID_AXIS(idim, com_cx)
	IF ( line .EQ. mnormal .OR. line .EQ. munknown ) THEN
	   modulo_axis = .FALSE.
	ELSE
	   axdim = TM_MODULO_LINE_DIM(line)   ! 2/02
	   modulo_axis = line_modulo(line)
     .		    .AND. hi_com-lo_com+1 .GE. axdim
	ENDIF
	IF ( modulo_axis ) axlen = TM_MODULO_AXLEN(line)

* it his a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO(line)
	IF (has_mod_void) THEN
	   has_mod_void = lo_com.LE.0 .OR. hi_com.GE.axdim
	ENDIF

* save the desired result region
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE

* get and save grid box coordinates just once for efficiency
	coord_lo_ss = MAX( lo_sub(idim)-edge, lo_com )
	coord_hi_ss = MIN( hi_sub(idim)+edge, hi_com )
        CALL MAKE_AXIS( mr_grid(com_mr), idim,
     .			coord_lo_ss, coord_hi_ss, coords )
        pos0    = coord_lo_ss - 1

* on action axis compute limits within which source data is accessible
	IF ( modulo_axis ) THEN
	   lo_edge = lo_sub(idim) - axdim	! complete axis wrap-around
	   hi_edge = hi_sub(idim) + axdim	! complete axis wrap-around
	ELSE
	   lo_edge = coord_lo_ss
	   hi_edge = coord_hi_ss
	ENDIF

* FILL ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
           no_lo_val = .TRUE.
           next_hi_ss = no_point
* ... find the closest off-scale low interpolation point in case we need it
           DO 101 i = lo_s1, lo_edge, -1
	      modulo = modulo_axis .AND. i.LT.coord_lo_ss
	      IF ( modulo ) THEN
	         look = MOD(i+axdim-coord_lo_ss,axdim) + coord_lo_ss
	      ELSE
	         look = i
	      ENDIF
              IF ( com(look,j,k,l) .NE. bad_res ) THEN
                 lo_val = com(look,j,k,l)
	         lo_index = look
	         lo_coord = coords(look-pos0)
	         IF ( modulo ) THEN
	            lo_index = lo_index - axdim
	            lo_coord = lo_coord - axlen
	         ENDIF
                 no_lo_val = .FALSE.
	         GOTO 102
              ENDIF
 101       CONTINUE
* ... process the requested line in X
 102	   DO 110 i = lo_s1, hi_s1
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
	         lo_index = i
	         lo_coord = coords(i-pos0)
                 no_lo_val = .FALSE.
	         GOTO 110
	      ENDIF
* ... need to fill a point
              IF (no_lo_val) THEN
* ... no lower valid point
                 res(i,j,k,l) = bad_res
	         GOTO 110
              ELSEIF ( next_hi_ss .LT. i ) THEN
* ... look forward from current position for next valid point
                 DO 104 ii = i+1, hi_edge
	            modulo = modulo_axis .AND. ii.GT.coord_hi_ss
	            IF ( modulo ) THEN
	               look = MOD(ii+axdim-coord_lo_ss,axdim) + coord_lo_ss
	            ELSE
	               look = ii
	            ENDIF
                    IF ( com(look,j,k,l) .NE. bad_res ) THEN
                       next_hi_ss = look
                       hi_val = com(look,j,k,l)
	               hi_index = look
	               hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) THEN
	                  hi_index = hi_index + (ii/axdim)*axdim
	                  hi_coord = hi_coord + (ii/axdim)*axlen
	               ENDIF
                       GOTO 108
                    ENDIF
 104             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 110
              ENDIF
* ... fill in interpolating value
 108	      coord    = coords(i-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
* ... bad if this interp crosses or lands on the void of a subspan modulo axis 
	      IF (has_mod_void) THEN
	        IF ( lo_index/axdim .NE. hi_index/axdim
     .	       .OR.   MOD(i,axdim)  .EQ. 0 )  res(i,j,k,l) = bad_res
	      ENDIF
 110	   CONTINUE

* FILL ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 i = lo_s1, hi_s1
           no_lo_val = .TRUE.
           next_hi_ss = no_point
* ... find the closest off-scale low interpolation point in case we need it
           DO 201 j = lo_s2, lo_edge, -1
	      modulo = modulo_axis .AND. j.LT.coord_lo_ss
              IF ( modulo ) THEN
                 look = MOD(j+axdim-coord_lo_ss,axdim) + coord_lo_ss
              ELSE
                 look = j
              ENDIF
              IF ( com(i,look,k,l) .NE. bad_res ) THEN
                 lo_val = com(i,look,k,l)
	         lo_index = look
	         lo_coord = coords(look-pos0)
	         IF ( modulo ) THEN
	            lo_index = lo_index - axdim
	            lo_coord = lo_coord - axlen
	         ENDIF
                 no_lo_val = .FALSE.
                 GOTO 202
              ENDIF
 201       CONTINUE
* ... process the requested line in Y
 202	   DO 210 j = lo_s2, hi_s2
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
	         lo_index = j
                 lo_coord = coords(j-pos0)
                 no_lo_val = .FALSE.
	         GOTO 210
	      ENDIF
* ... need to fill a point
              IF (no_lo_val) THEN
* ... no lower valid point
                 res(i,j,k,l) = bad_res
                 GOTO 210
              ELSEIF ( next_hi_ss .LT. j ) THEN
* ... look forward from current position for next valid point
                 DO 204 jj = j+1, hi_edge
                    modulo = modulo_axis .AND. jj.GT.coord_hi_ss
                    IF ( modulo ) THEN
		       look = MOD(jj+axdim-coord_lo_ss,axdim)+coord_lo_ss
                    ELSE
                       look = jj
                    ENDIF
                    IF ( com(i,look,k,l) .NE. bad_res ) THEN
                       next_hi_ss = look
                       hi_val = com(i,look,k,l)
	               hi_index = look
		       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) THEN
	                  hi_index = hi_index + (jj/axdim)*axdim
	                  hi_coord = hi_coord + (jj/axdim)*axlen
	               ENDIF
                       GOTO 208
                    ENDIF
 204             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 210
              ENDIF
* ... fill in interpolating value
 208          coord    = coords(j-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
* ... bad if this interp crosses or lands on the void of a subspan modulo axis 
	      IF (has_mod_void) THEN
	        IF ( lo_index/axdim .NE. hi_index/axdim
     .	       .OR.   MOD(j,axdim)  .EQ. 0 )  res(i,j,k,l) = bad_res
	      ENDIF
 210	   CONTINUE

* FILL ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
           no_lo_val = .TRUE.
           next_hi_ss = no_point
* ... find the closest off-scale low interpolation point in case we need it
           DO 301 k = lo_s3, lo_edge, -1
              modulo = modulo_axis .AND. k.LT.coord_lo_ss
              IF ( modulo ) THEN
                 look = MOD(k+axdim-coord_lo_ss,axdim) + coord_lo_ss
              ELSE
                 look = k
              ENDIF
              IF ( com(i,j,look,l) .NE. bad_res ) THEN
                 lo_val = com(i,j,look,l)
	         lo_index = look
	         lo_coord = coords(look-pos0)
	         IF ( modulo ) THEN
	            lo_index = lo_index - axdim
	            lo_coord = lo_coord - axlen
	         ENDIF
                 no_lo_val = .FALSE.
	         GOTO 302
              ENDIF
 301       CONTINUE
* ... process the requested line in Z
 302	   DO 310 k = lo_s3, hi_s3
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
	         lo_index = k
                 lo_coord = coords(k-pos0)
                 no_lo_val = .FALSE.
	         GOTO 310
              ENDIF
* ... need to fill a point
              IF (no_lo_val) THEN
* ... no lower valid point
                 res(i,j,k,l) = bad_res
                 GOTO 310
              ELSEIF ( next_hi_ss .LT. k ) THEN
* ... look forward from current position for next valid point
                 DO 304 kk = k+1, hi_edge
                    modulo = modulo_axis .AND. kk.GT.coord_hi_ss
                    IF ( modulo ) THEN
		       look = MOD(kk+axdim-coord_lo_ss,axdim) + coord_lo_ss
                    ELSE
                       look = kk
                    ENDIF
                    IF ( com(i,j,look,l) .NE. bad_res ) THEN
                       next_hi_ss = kk
                       hi_val = com(i,j,look,l)
	               hi_index = look
                       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) THEN
	                  hi_index = hi_index + (kk/axdim)*axdim
	                  hi_coord = hi_coord + (kk/axdim)*axlen
	               ENDIF
                       GOTO 308
                    ENDIF
 304             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 310
              ENDIF
* ... fill in interpolating value
 308          coord    = coords(k-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
* ... bad if this interp crosses or lands on the void of a subspan modulo axis 
	      IF (has_mod_void) THEN
	        IF ( lo_index/axdim .NE. hi_index/axdim
     .	       .OR.   MOD(k,axdim)  .EQ. 0 )  res(i,j,k,l) = bad_res
	      ENDIF
 310	   CONTINUE

* FILL ALONG T AXIS
	ELSE
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
           no_lo_val = .TRUE.
           next_hi_ss = no_point
* ... find the closest off-scale low interpolation point in case we need it
           DO 401 l = lo_s4, lo_edge, -1
              modulo = modulo_axis .AND. l.LT.coord_lo_ss
              IF ( modulo ) THEN
                 look = MOD(l+axdim-coord_lo_ss,axdim) + coord_lo_ss
              ELSE
                 look = l
              ENDIF
              IF ( com(i,j,k,look) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,look)
	         lo_index = look
	         lo_coord = coords(look-pos0)
	         IF ( modulo ) THEN
	            lo_index = lo_index - axdim
	            lo_coord = lo_coord - axlen
	         ENDIF
                 no_lo_val = .FALSE.
	         GOTO 402
              ENDIF
 401       CONTINUE
* ... process the requested line in T
 402	   DO 410 l = lo_s4, hi_s4
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
	         lo_index = l
                 lo_coord = coords(l-pos0)	! fixed 8/95
                 no_lo_val = .FALSE.
	         GOTO 410
	      ENDIF
* ... need to fill a point
              IF (no_lo_val) THEN
* ... no lower valid point
                 res(i,j,k,l) = bad_res
                 GOTO 410
              ELSEIF ( next_hi_ss .LT. l ) THEN
* ... look forward from current position for next valid point
                 DO 404 ll = l+1, hi_edge
                    modulo = modulo_axis .AND. ll.GT.coord_hi_ss
                    IF ( modulo ) THEN
		       look = MOD(ll+axdim-coord_lo_ss,axdim) + coord_lo_ss
                    ELSE
                       look = ll
                    ENDIF
                    IF ( com(i,j,k,look) .NE. bad_res ) THEN
                       next_hi_ss = look
                       hi_val = com(i,j,k,look)
	               hi_index = look
                       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) THEN
	                  hi_index = hi_index + (ll/axdim)*axdim
	                  hi_coord = hi_coord + (ll/axdim)*axlen
	               ENDIF
                       GOTO 408
                    ENDIF
 404             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 410
              ENDIF
* ... fill in interpolating value
 408          coord    = coords(l-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
* ... bad if this interp crosses or lands on the void of a subspan modulo axis 
	      IF (has_mod_void) THEN
	        IF ( lo_index/axdim .NE. hi_index/axdim
     .	       .OR.   MOD(l,axdim)  .EQ. 0 )  res(i,j,k,l) = bad_res
	      ENDIF
 410	   CONTINUE
	ENDIF

* always success
	DO_FILL_INTERP = ferr_ok
	RETURN

	END	
