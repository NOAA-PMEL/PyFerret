	INTEGER FUNCTION DO_FILL_INTERP(idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					coords	)

* fill along axis idim by interpolating from the closest available valid pt

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  3/23/92 - based on DO_FILL_AVE
* V301:  2/15/94 - upgrade to accomodate modulo axes
* V402:   8/9/95 - bug fix: T=@FLN had "lo_coord = coords(k-pos0)"

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
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
	REAL	arg, coords(*),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	LOGICAL	modulo_axis, modulo, no_lo_val
	INTEGER i, j, k, l, ii, jj, kk, ll, lo_sub(4), hi_sub(4),
     .		ii_com, lo_edge, hi_edge, pos0, edge,
     .		lo_com, hi_com, line, grid, axdim, look,
     .		coord_lo_ss, coord_hi_ss, next_hi_ss
	REAL	bad_res, lo_coord, hi_coord, coord, lo_val, hi_val,
     .		axlen
	REAL*8	TM_WORLD

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

* limits of the component data provided
	lo_com = cx_lo_ss(com_cx, idim)
	hi_com = cx_hi_ss(com_cx, idim)

* is it a modulo axis with sufficient component span provided to wrap around
	grid = cx_grid(com_cx)
	line = grid_line( idim, grid )
	IF ( line .EQ. mnormal .OR. line .EQ. munknown ) THEN
	   modulo_axis = .FALSE.
	ELSE
	   axdim = line_dim(line)
	   modulo_axis = line_modulo(line)
     .		    .AND. hi_com-lo_com+1 .GE. axdim
	ENDIF
	IF ( modulo_axis ) axlen = TM_WORLD(axdim,grid,idim,box_hi_lim)
     .				 - TM_WORLD( 1,   grid,idim,box_lo_lim)

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
	         lo_coord = coords(look-pos0)
	         IF ( modulo ) lo_coord = lo_coord - axlen
                 no_lo_val = .FALSE.
	         GOTO 102
              ENDIF
 101       CONTINUE
* ... process the requested line in X
 102	   DO 110 i = lo_s1, hi_s1
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
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
	               hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) hi_coord = hi_coord + (ii/axdim)*axlen
                       GOTO 108
                    ENDIF
 104             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 110
              ENDIF
* ... fill in interpolating value
 108          coord    = coords(i-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
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
                 lo_coord = coords(look-pos0)
                 IF ( modulo ) lo_coord = lo_coord - axlen
                 no_lo_val = .FALSE.
                 GOTO 202
              ENDIF
 201       CONTINUE
* ... process the requested line in Y
 202	   DO 210 j = lo_s2, hi_s2
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
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
		       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) hi_coord = hi_coord + (jj/axdim)*axlen
                       GOTO 208
                    ENDIF
 204             CONTINUE
* ... no higher valid point
                 res(i,j,k,l) = bad_res
                 GOTO 210
              ENDIF
* ... fill in interpolating value
 208           coord    = coords(j-pos0)
              res(i,j,k,l) = ( hi_val*(coord-lo_coord)
     .                     +   lo_val*(hi_coord-coord) )
     .                     / (hi_coord - lo_coord)
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
                 lo_coord = coords(look-pos0)
                 IF ( modulo ) lo_coord = lo_coord - axlen
                 no_lo_val = .FALSE.
	         GOTO 302
              ENDIF
 301       CONTINUE
* ... process the requested line in Z
 302	   DO 310 k = lo_s3, hi_s3
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
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
                       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) hi_coord = hi_coord + (kk/axdim)*axlen
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
                 lo_coord = coords(look-pos0)
                 IF ( modulo ) lo_coord = lo_coord - axlen
                 no_lo_val = .FALSE.
	         GOTO 402
              ENDIF
 401       CONTINUE
* ... process the requested line in T
 402	   DO 410 l = lo_s4, hi_s4
	      IF ( com(i,j,k,l) .NE. bad_res ) THEN
	         res(i,j,k,l) = com(i,j,k,l)
                 lo_val = com(i,j,k,l)
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
                       hi_coord = coords(next_hi_ss-pos0)
		       IF ( modulo ) hi_coord = hi_coord + (ll/axdim)*axlen
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
 410	   CONTINUE
	ENDIF

* always success
	DO_FILL_INTERP = ferr_ok
	RETURN

	END	
