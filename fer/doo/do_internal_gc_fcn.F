      SUBROUTINE DO_INTERNAL_GC_FCN( memory,
     .				fcn, num_com, com_cx,
     .				com1, mr1,
     .				com2, mr2,
     .				com3, mr3,
     .				com4, mr4,
     .				com5, mr5,
     .				com6, mr6,
     .				com7, mr7,
     .				com8, mr8,
     .				com9, mr9,
     .				res, mres,
     .                          status )

* execute a grid changing function

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

* V450:  1/97 - new, based on operate.F, but for grid changing functions
*        7/97 - complete the ZAXREPLACE function
*       7/25/97 - increase number of args from 4 to 9
*               - add function PPLGRID

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	fcn, num_com, com_cx(9), status,
     .		mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mres
        REAL	memory( mem_blk_size, max_mem_blks )
	REAL    com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		com4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .		com5( m5lox:m4hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit ),
     .		com6( m6lox:m4hix,m6loy:m6hiy,m6loz:m6hiz,m6lot:m6hit ),
     .		com7( m7lox:m4hix,m7loy:m7hiy,m7loz:m7hiz,m7lot:m7hit ),
     .		com8( m8lox:m4hix,m8loy:m8hiy,m8loz:m8hiz,m8lot:m8hit ),
     .		com9( m9lox:m4hix,m9loy:m9hiy,m9loz:m9hiz,m9lot:m9hit ),
     .		 res( m10lox:m10hix,m10loy:m10hiy,
     .					  m10loz:m10hiz,m10lot:m10hit )

* internal variable declarations:
	LOGICAL	first
	INTEGER CX_DIM_LEN,  CGRID_SIZE, count, size,
     .		i, j, k, l, icom, idim,
     .		cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9,
     .		i1, j1, k1, l1, i2, j2, k2, l2,
     .		i3, j3, k3, l3, i4, j4, k4, l4,
     .		i5, j5, k5, l5, i6, j6, k6, l6,
     .		di1, dj1, dk1, dl1, di2, dj2, dk2, dl2,
     .		di3, dj3, dk3, dl3, di4, dj4, dk4, dl4,
     .		di5, dj5, dk5, dl5, di6, dj6, dk6, dl6,
     .		si1, sj1, sk1, sl1, si2, sj2, sk2, sl2,
     .		si3, sj3, sk3, sl3, si4, sj4, sk4, sl4,
     .		si5, sj5, sk5, sl5, si6, sj6, sk6, sl6
        INTEGER nz, ndim, ndx_lo, wkblk

	REAL	RANDU, RANDN, RHO_UNESCO, THETA_FOFF,
     .		DAYS_FROM_DAY0, bad_res,
     .          bad1, bad2, bad3, bad4, bad5, bad6, bad7, bad8, bad9
        REAL    zval, frac

        REAL*8  TM_WORLD

* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(4,6), lo(4,6), ind(4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1),
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2),
     .	(lo(1,3),si3), (lo(2,3),sj3), (lo(3,3),sk3), (lo(4,3),sl3),
     .	(lo(1,4),si4), (lo(2,4),sj4), (lo(3,4),sk4), (lo(4,4),sl4),
     .	(lo(1,5),si5), (lo(2,5),sj5), (lo(3,5),sk5), (lo(4,5),sl5),
     .	(lo(1,6),si6), (lo(2,6),sj6), (lo(3,6),sk6), (lo(4,6),sl6),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),
     .	(inc(1,3),di3),(inc(2,3),dj3),(inc(3,3),dk3),(inc(4,3),dl3),
     .	(inc(1,4),di4),(inc(2,4),dj4),(inc(3,4),dk4),(inc(4,4),dl4),
     .	(inc(1,5),di5),(inc(2,5),dj5),(inc(3,5),dk5),(inc(4,5),dl5),
     .	(inc(1,6),di6),(inc(2,6),dj6),(inc(3,6),dk6),(inc(4,6),dl6),
     .	(ind(1),i2),   (ind(2),j2),   (ind(3),k2),   (ind(4),l2)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )

* initialize commonly used variables
        status = ferr_ok
	cx1 = com_cx(1)
	cx2 = com_cx(2)
	cx3 = com_cx(3)
	cx4 = com_cx(4)
	cx5 = com_cx(5)
	cx6 = com_cx(6)
	cx7 = com_cx(7)
	cx8 = com_cx(8)
	cx9 = com_cx(9)

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad4 = mr_bad_data ( mr4 )
	bad5 = mr_bad_data ( mr5 )
	bad6 = mr_bad_data ( mr6 )
	bad7 = mr_bad_data ( mr7 )
	bad8 = mr_bad_data ( mr8 )
	bad9 = mr_bad_data ( mr9 )
	bad_res = mr_bad_data( mres )

* compute the delta increment for each axis of each component
	DO 10 icom = 1, num_com
	DO 10 idim = 1, 4
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
 10	CONTINUE

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
* NOTE: 
	DO 20 icom = 1, num_com
	DO 20 idim = 1, 4
 20	lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)

* select the function
	GOTO (  1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
     .		2100,2200,2300,2400,2500,2600,2700,2800,2900,3000
     .								  ) fcn

* "UNRAVEL" - reshape/flatten the input into a line in X
 1100	i2 = m5lox
	DO 1110 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	 DO 1110 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO 1110 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	   DO 1110 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i2,m5loy,m5loz,m5lot) = bad_res
	   ELSE
	      res(i2,m5loy,m5loz,m5lot) = com1(i,j,k,l)
	   ENDIF
	   i2 = i2 + 1

 1110	CONTINUE
	RETURN

* "TIMES2" - debugging function
 1200	DO 1210 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 1210 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 1210 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 1210 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .    .OR.  com1(i,j,k,l) .LE. 0.0 ) THEN    ! 8/92
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i,j,k,l) * 2.0
	   ENDIF
 1210	CONTINUE
	RETURN

* "RESHAPE" - reshape argument 1 into the grid of arg 2. Start output
* at the i,j,k,l point given in arg 2. It is commonly useful for taking
* monthly data and recasting it as a 12 month Z axis by a yearly axis
 1300	DO 1310 idim = 1, 4
 1310	ind(idim) = cx_lo_ss(cx1,idim) !i2,j2,k2,l2 <-- comp 1 start indices
	size = CGRID_SIZE(cx1)
	count = 0	! total points xferred - used to prevent grid overflow

	DO 1350 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 1350 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 1350 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 1350 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   count = count + 1
	   IF ( count .GT. size ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( com1(i2,j2,k2,l2) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i2,j2,k2,l2)
	   ENDIF

* ... increment the subscripts of the input argument
	   DO 1320 idim = 1, 4
	      IF (ind(idim) .LT. cx_hi_ss(cx1,idim) ) THEN
	         ind(idim) = ind(idim) + 1
	         GOTO 1350
	      ELSE
	         ind(idim) =  cx_lo_ss(cx1,idim)
	      ENDIF
 1320	   CONTINUE

 1350	CONTINUE
	RETURN

* "ZAXREPLACE(V,ZVALS,ZAX)" - regrid a sigma-coordinate-style (layered)
* variable, V, onto a depth axis, ZAX, guided by the depths in ZVALS
* ... sanity check -- V and ZVALS must match in vertical structure
 1400   IF ( (cx_lo_s3(cx1) .NE. cx_lo_s3(cx2))
     . .OR.  (cx_hi_s3(cx1) .NE. cx_hi_s3(cx2)) ) CALL ERRMSG
     .		(ferr_limits, status,
     .		 'Z limits of data and depth fields must match', *5000)

* ... allocate a work array for a single profile of Z depths
	CALL GET_WORK_SPC( m2hiz-m2loz+1, wkblk, status )
        IF ( status .NE. ferr_ok ) GOTO 5000

* ... and creating a context to describe a single profile
	CALL TRANSFER_CONTEXT( cx2, cx_buff )

!* ... now extract the desired vertical axis points in single prec.
!	CALL MAKE_AXIS(mr_grid(mr3), z_dim, sk2, hiz, memory(1,wkblk))
!  (not used because only one work buffer was easily available ...
!             TM_WORLD calls are needed repeatedly, instead)

* ... loop over the X,Y,T range of the input fields
	l1 = sl1
	l2 = sl2
	DO 1450 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	cx_lo_s4(cx_buff) = l2
	cx_hi_s4(cx_buff) = l2
	j1 = sj1
	j2 = sj2
	 DO 1450 j = mr_lo_s2(mres), mr_hi_s2(mres)
	 j1 = j1 + dj1
	 j2 = j2 + dj2
	 cx_lo_s2(cx_buff) = j2 
	 cx_hi_s2(cx_buff) = j2 
	 i1 = si1
	 i2 = si2
	  DO 1450 i = mr_lo_s1(mres), mr_hi_s1(mres)
	  i1 = i1 + di1
	  i2 = i2 + di2
	  cx_lo_s1(cx_buff) = i2 
	  cx_hi_s1(cx_buff) = i2 

* ... extract the vertical profile at this I,J,L location
           CALL EXTRACT_LINE ( cx_buff,
     .                      com2,
     .                      mr2,
     .                      memory(1,wkblk),
     .                      idim,	! unused
     .                      ndim,	! unused
     .                      nz,
     .                      status      )
           IF ( status .NE. ferr_ok ) THEN
              CALL RELEASE_WORK_SPC
              GOTO 5000
           ENDIF

* loop over the desired output Z-axis points
          ndx_lo = 0    ! no guess for location of first depth
	  DO 1410 k = mr_lo_s3(mres), mr_hi_s3(mres)

* ... next Z axis value (com3) to locate
           zval = SNGL(TM_WORLD(k,cx_grid(cx3),z_dim,box_middle))

* ... search the Z field (com2) for the index just below this value
	   CALL HUNT_R4(memory(1,wkblk), 1, nz, zval, ndx_lo, frac)

           IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
              res(i,j,k,l) = bad_res
           ELSEIF ( frac .EQ. 1.0 ) THEN
	      res(i,j,k,l) =  com1(i1,j1,sk1+ndx_lo,l1)
           ELSE
* ... interpolate data com1 field to get result
	      res(i,j,k,l) =  frac   * com1(i1,j1,sk1+ndx_lo,  l1)
     .			   + (1-frac)* com1(i1,j1,sk1+ndx_lo+1,l1)
           ENDIF

 1410	  CONTINUE
 1450	CONTINUE
        CALL RELEASE_WORK_SPC
        RETURN

 1500	CONTINUE
 1600	CONTINUE
 1700	CONTINUE
 1800	CONTINUE
 1900	CONTINUE
 2000	CONTINUE
 2100	CONTINUE
 2200	CONTINUE
 2300	CONTINUE
 2400	CONTINUE
 2500	CONTINUE
 2600	CONTINUE
 2700	CONTINUE
 2800	CONTINUE
 2900	CONTINUE
 3000	CONTINUE
	STOP 'GC FUNCTION NOT IMPLEMENTED'


! helpful for editing ...
! 1400	l1 = sl1
!	l2 = sl2
!	DO 1410 l = mr_lo_s4(mres), mr_hi_s4(mres)
!	l1 = l1 + dl1
!	l2 = l2 + dl2
!	k1 = sk1
!	k2 = sk2
!	 DO 1410 k = mr_lo_s3(mres), mr_hi_s3(mres)
!	 k1 = k1 + dk1
!	 k2 = k2 + dk2
!	 j1 = sj1
!	 j2 = sj2
!	  DO 1410 j = mr_lo_s2(mres), mr_hi_s2(mres)
!	  j1 = j1 + dj1
!	  j2 = j2 + dj2
!	  i1 = si1
!	  i2 = si2
!	   DO 1410 i = mr_lo_s1(mres), mr_hi_s1(mres)
!	   i1 = i1 + di1
!	   i2 = i2 + di2
!
!	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
!     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
!	      res(i,j,k,l) = bad_res
!	   ELSE
!	      res(i,j,k,l) = com1(i1,j1,k1,l1) + com2(i2,j2,k2,l2)
!	   ENDIF
!
! 1410	CONTINUE

* error exits
 5000	RETURN
	END

