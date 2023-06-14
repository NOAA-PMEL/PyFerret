*
* samplexy_curv_avg.F
*
* Ansley Manke
* 4/03 
* ???  Note: does not currently deal with modulo data  ???
* 1/06 Remove calls to SIGN to compile with f90 on IRIX.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* This function samples 4-d curvlinear data on the x and y axes 
* indicated by args 4 and 5
*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE samplexy_curv_avg_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a set of (X,Y) points, ',
     .        'using unweighted averaging')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 5)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        NORMAL,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t,e,f) to sample')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'DAT_LON')
      CALL ef_set_arg_desc(id, arg, 'longitudes of input variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'DAT_LAT')
      CALL ef_set_arg_desc(id, arg, 'latitudes of input variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 'X values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE samplexy_curv_avg_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER my_lo, my_hi
      INTEGER nx, ny, nz, nt, ne, nf

*     Use utility functions to get context information about the 
*     fourth argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG4) - arg_lo_ss(X_AXIS, ARG4) + 1
      ny = arg_hi_ss(Y_AXIS, ARG4) - arg_lo_ss(Y_AXIS, ARG4) + 1
      nz = arg_hi_ss(Z_AXIS, ARG4) - arg_lo_ss(Z_AXIS, ARG4) + 1
      nt = arg_hi_ss(T_AXIS, ARG4) - arg_lo_ss(T_AXIS, ARG4) + 1
      ne = arg_hi_ss(E_AXIS, ARG4) - arg_lo_ss(E_AXIS, ARG4) + 1
      nf = arg_hi_ss(F_AXIS, ARG4) - arg_lo_ss(F_AXIS, ARG4) + 1

      my_lo = 1
      my_hi = max(nx,ny,nz,nt,ne,nf)

      CALL ef_set_axis_limits(id, X_AXIS, my_lo, my_hi)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE samplexy_curv_avg_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER nx, ny
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  workspace for saving locations of xpt in arg2 array.

      nx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) 
      ny = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) 

*  xlocations
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                            nx*ny, 2, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



*
* In this subroutine we compute the result
*
      SUBROUTINE samplexy_curv_avg_compute(id, arg_1, arg_2, arg_3,
     .                             arg_4, arg_5, result, xlocations)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .           mem5lot:mem5hit, mem5loe:mem5hie, mem5lof:mem5hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

      REAL xlocations(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .                wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .                wrk1loe:wrk1hie, wrk1lof:wrk1hif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER nx, nxx, nxy, nxz, nxt, nxe, nxf
      INTEGER ny, nyx, nyy, nyz, nyt, nye, nyf
      INTEGER ndimx, ndimy
      INTEGER nx1, nx2, nx3, ny1, ny2, ny3
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER i4, j4, k4, l4, m4, n4
      INTEGER i5, j5, k5, l5, m5, n5
      INTEGER ibot, itop, jbot, jtop
      INTEGER nloc, iloc, ptloc, icross
      LOGICAL need_modulo
      REAL ylocations(2)
      REAL xpt, ypt, ydiff, yd
      REAL xptmin, xptmax, xcoordmin, xcoordmax
      REAL fbb, ftb, fbt, ftt
      REAL sum, count
      REAL add_xcoord, xadd, xcoord
      REAL xcoord_diff, xcoordlast, tol
      CHARACTER*255 err_msg
      parameter (tol = 140.)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C Check that first 3 args are on the same grid.

      nx1 = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      nx2 = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nx3 = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      ny1 = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1
      ny2 = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      ny3 = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      IF ( (nx1 .NE. nx2) .OR. (nx1 .NE. nx3) .OR.
     .     (ny1 .NE. ny2) .OR. (ny1 .NE. ny3) ) THEN
         err_msg ='Arguments 1, 2, and 3 must have the same XY grid'
         GO TO 999
      ENDIF

      IF ( (arg_hi_ss(Z_AXIS,ARG2) .GT. arg_lo_ss(Z_AXIS,ARG2)) .OR.
     .     (arg_hi_ss(T_AXIS,ARG2) .GT. arg_lo_ss(T_AXIS,ARG2)) .OR.
     .     (arg_hi_ss(E_AXIS,ARG2) .GT. arg_lo_ss(E_AXIS,ARG2)) .OR.
     .     (arg_hi_ss(F_AXIS,ARG2) .GT. arg_lo_ss(F_AXIS,ARG2)) .OR.
     .     (arg_hi_ss(Z_AXIS,ARG3) .GT. arg_lo_ss(Z_AXIS,ARG3)) .OR.
     .     (arg_hi_ss(T_AXIS,ARG3) .GT. arg_lo_ss(T_AXIS,ARG3)) .OR.
     .     (arg_hi_ss(E_AXIS,ARG3) .GT. arg_lo_ss(E_AXIS,ARG3)) .OR.
     .     (arg_hi_ss(F_AXIS,ARG3) .GT. arg_lo_ss(F_AXIS,ARG3)) ) THEN
         err_msg ='Arguments 2 and 3 must be defined only on X and Y'
         GO TO 999
      ENDIF

C Check that sample x and y are simple lists of points - same length

      nxx = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      nxy = arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4) + 1
      nxz = arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4) + 1
      nxt = arg_hi_ss(T_AXIS,ARG4) - arg_lo_ss(T_AXIS,ARG4) + 1
      nxe = arg_hi_ss(E_AXIS,ARG4) - arg_lo_ss(E_AXIS,ARG4) + 1
      nxf = arg_hi_ss(F_AXIS,ARG4) - arg_lo_ss(F_AXIS,ARG4) + 1
      nx = max(nxx, nxy, nxz, nxt, nxe, nxf)

      nyx = arg_hi_ss(X_AXIS,ARG5) - arg_lo_ss(X_AXIS,ARG5) + 1
      nyy = arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5) + 1
      nyz = arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5) + 1
      nyt = arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5) + 1
      nye = arg_hi_ss(E_AXIS,ARG5) - arg_lo_ss(E_AXIS,ARG5) + 1
      nyf = arg_hi_ss(F_AXIS,ARG5) - arg_lo_ss(F_AXIS,ARG5) + 1
      ny = max(nyx, nyy, nyz, nyt, nye, nyf)

      ndimx = 0
      ndimy = 0
      DO i = X_AXIS,F_AXIS
         IF ( arg_hi_ss(i,ARG4) .GT. arg_lo_ss(i,ARG4) ) ndimx=ndimx + 1
         IF ( arg_hi_ss(i,ARG5) .GT. arg_lo_ss(i,ARG5) ) ndimy=ndimy + 1
      ENDDO

      IF ( (nx .NE. ny) .OR. (ndimx .GT. 1) .OR. (ndimy .GT.1) ) THEN
         err_msg = 
     .  'Arguments 4 and 5 must be 1-dimensional lists of equal length'
         GO TO 999
      ENDIF
      
* Initialize result

      DO n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         result(i,j,k,l,m,n) = bad_flag_result
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

* Get range of arg_2, longitude coordinates, and arg_4, xpts to sample.
* is modulo-ing needed?  Also checking for bad values.

*     Z/T/E/F axes of arg_2 are normal or singleton
      n2 = arg_lo_ss(F_AXIS,ARG2)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)

      xcoordmin = ABS(bad_flag(ARG2))
      xcoordmax = -1. * xcoordmin

      DO j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      DO i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
         IF (arg_2(i2,j2,k2,l2,m2,n2) .NE. bad_flag(ARG2)) THEN
            xcoordmin = MIN(arg_2(i2,j2,k2,l2,m2,n2), xcoordmin)
            xcoordmax = MAX(arg_2(i2,j2,k2,l2,m2,n2), xcoordmax)
         ELSE
            err_msg =
     .      'Arg 2, curvilinear X coordinates, contains missing values'
            GOTO 999
         ENDIF
      ENDDO
      ENDDO

      xptmin = ABS(bad_flag(ARG4))
      xptmax = -1.* xptmin

      DO n4 = arg_lo_ss(F_AXIS,ARG4), arg_hi_ss(F_AXIS,ARG4)
      DO m4 = arg_lo_ss(E_AXIS,ARG4), arg_hi_ss(E_AXIS,ARG4)
      DO l4 = arg_lo_ss(T_AXIS,ARG4), arg_hi_ss(T_AXIS,ARG4)
      DO k4 = arg_lo_ss(Z_AXIS,ARG4), arg_hi_ss(Z_AXIS,ARG4)
      DO j4 = arg_lo_ss(Y_AXIS,ARG4), arg_hi_ss(Y_AXIS,ARG4)
      DO i4 = arg_lo_ss(X_AXIS,ARG4), arg_hi_ss(X_AXIS,ARG4)
         IF (arg_4(i4,j4,k4,l4,m4,n4) .NE. bad_flag(ARG4)) THEN
            xptmin = MIN(arg_4(i4,j4,k4,l4,m4,n4), xptmin)
            xptmax = MAX(arg_4(i4,j4,k4,l4,m4,n4), xptmax)
         ELSE
            err_msg =
     .      'Arg 4, sample X points, contains missing values'
            GOTO 999
         ENDIF
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      IF (xptmin .LT. xcoordmin) THEN
         need_modulo = .TRUE.
         add_xcoord = -360.
      ENDIF
      IF (xptmax .GT. xcoordmax) THEN
         need_modulo = .TRUE.
         add_xcoord = 360.
      ENDIF

* Check once for bad values so we do not have to worry about this issue

*     Z/T/E/F axes of arg_3 are normal or singleton
      n3 = arg_lo_ss(F_AXIS,ARG2)
      m3 = arg_lo_ss(E_AXIS,ARG2)
      l3 = arg_lo_ss(T_AXIS,ARG2)
      k3 = arg_lo_ss(Z_AXIS,ARG2)

      DO j3 = arg_lo_ss(Y_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3)
      DO i3 = arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3)
         IF (arg_3(i3,j3,k3,l3,m3,n3) .EQ. bad_flag(ARG3)) THEN
            err_msg =
     .      'Arg 3, curvilinear Y coordinates, contains missing values'
            GOTO 999
         ENDIF
      ENDDO
      ENDDO

      DO n5 = arg_lo_ss(F_AXIS,ARG5), arg_hi_ss(F_AXIS,ARG5)
      DO m5 = arg_lo_ss(E_AXIS,ARG5), arg_hi_ss(E_AXIS,ARG5)
      DO l5 = arg_lo_ss(T_AXIS,ARG5), arg_hi_ss(T_AXIS,ARG5)
      DO k5 = arg_lo_ss(Z_AXIS,ARG5), arg_hi_ss(Z_AXIS,ARG5)
      DO j5 = arg_lo_ss(Y_AXIS,ARG5), arg_hi_ss(Y_AXIS,ARG5)
      DO i5 = arg_lo_ss(X_AXIS,ARG5), arg_hi_ss(X_AXIS,ARG5)
         IF (arg_5(i5,j5,k5,l5,m5,n5) .EQ. bad_flag(ARG5)) THEN
            err_msg =
     .      'Arg 5, sample Y points, contains missing values'
            GOTO 999
         ENDIF
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

*  For each (xpt,ypt) pair, search the data arrays arg2,arg3
*  for the nearest higher (x,y) lat-lon coordinates.  Interpolate 
*  arg_1 in 2 directions for the result.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      j4 = arg_lo_ss(Y_AXIS,ARG4)
      k4 = arg_lo_ss(Z_AXIS,ARG4)
      l4 = arg_lo_ss(T_AXIS,ARG4)
      m4 = arg_lo_ss(E_AXIS,ARG4)
      n4 = arg_lo_ss(F_AXIS,ARG4)

      i5 = arg_lo_ss(X_AXIS,ARG5)
      j5 = arg_lo_ss(Y_AXIS,ARG5)
      k5 = arg_lo_ss(Z_AXIS,ARG5)
      l5 = arg_lo_ss(T_AXIS,ARG5)
      m5 = arg_lo_ss(E_AXIS,ARG5)
      n5 = arg_lo_ss(F_AXIS,ARG5)

* Y-axis is normal to the result
      j = res_lo_ss(Y_AXIS)

      DO 700 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

* Find all locations of xpt in arg_2, longitudes of data
* When a change of sign corresponds with crossing a branch cut, that
* is not saved as a location where xpt-minus-coordinate changes sign.

         xadd = 0.
         xpt = arg_4(i4,j4,k4,l4,m4,n4)
         IF (need_modulo .AND. 
     .       ((xpt .LT. xcoordmin) .OR.(xpt .GT. xcoordmax)) )
     .      xadd = add_xcoord

         nloc = 0
         i2 = arg_lo_ss(X_AXIS,ARG2)
	 j2 = arg_lo_ss(Y_AXIS,ARG2)
	 xcoordlast = arg_2(i2,j2,k2,l2,m2,n2) + xadd

         DO 200 j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
            i2 = arg_lo_ss(X_AXIS,ARG2)
            xcoord = arg_2(i2,j2,k2,l2,m2,n2) + xadd
            icross = 1
            IF ((xpt - xcoord) .LT. 0.) icross = -1

            DO 100 i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
               xcoord = arg_2(i2,j2,k2,l2,m2,n2) + xadd
	       xcoord_diff = ABS(xcoordlast-xcoord)
	       IF (xcoord_diff .GE. tol) icross = -1*icross

               iloc = 1
               IF ((xpt-xcoord) .LT. 0. ) iloc = -1

               IF (icross .NE. iloc)THEN
                  nloc = nloc + 1
                  xlocations(nloc,1,1,1,1,1) =
     .                       MAX(i2-1,arg_lo_ss(X_AXIS,ARG2))
                  xlocations(nloc,2,1,1,1,1) = j2
                  icross = 1
                  IF ((xpt - xcoord) .LT. 0.) icross = -1
               ENDIF
	       xcoordlast = xcoord 
  100       CONTINUE

  200    CONTINUE

* Which of those locations is nearest ypt in arg_3, latitudes of data?

         ypt = arg_5(i5,j5,k5,l5,m5,n5)
         ydiff = ABS(bad_flag_result)
         ptloc = 1

         DO iloc = 1, nloc
            i3 = xlocations(iloc,1,1,1,1,1)
            j3 = xlocations(iloc,2,1,1,1,1)

            yd = ABS(ypt - arg_3(i3,j3,k3,l3,m3,n3))
            IF ( yd .LT. ydiff ) THEN
               ylocations(1) = i3
               ylocations(2) = j3
               ydiff = yd
               ptloc = iloc
            ENDIF
         ENDDO

         ibot = xlocations(ptloc,1,1,1,1,1)
         jbot = xlocations(ptloc,2,1,1,1,1)
         IF (ibot .EQ. arg_hi_ss(X_AXIS,ARG1) ) ibot = ibot - 1
         IF (jbot .EQ. arg_hi_ss(Y_AXIS,ARG1) ) jbot = jbot - 1

         itop = ibot + 1
         jtop = jbot + 1

         IF ( ibot .GE. arg_lo_ss(X_AXIS,ARG1) .AND.
     .        itop .LE. arg_hi_ss(X_AXIS,ARG1) .AND.
     .        jbot .GE. arg_lo_ss(Y_AXIS,ARG1) .AND.
     .        jtop .LE. arg_hi_ss(Y_AXIS,ARG1) ) THEN

            n1 = arg_lo_ss(F_AXIS,ARG1)
            DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

            m1 = arg_lo_ss(E_AXIS,ARG1)
            DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  fbb = arg_1(ibot,jbot,k1,l1,m1,n1)
                  ftb = arg_1(itop,jbot,k1,l1,m1,n1)
                  fbt = arg_1(ibot,jtop,k1,l1,m1,n1)
                  ftt = arg_1(itop,jtop,k1,l1,m1,n1)

                  count = 0.
                  sum = 0.
                  IF (fbb .NE. bad_flag(ARG1)) THEN
                     count = count + 1.
                     sum = sum + fbb
                  ENDIF
                  IF (ftb .NE. bad_flag(ARG1)) THEN
                     count = count + 1.
                     sum = sum + ftb
                  ENDIF
                  IF (fbt .NE. bad_flag(ARG1)) THEN
                     count = count + 1.
                     sum = sum + fbt
                  ENDIF
                  IF (ftt .NE. bad_flag(ARG1)) THEN
                     count = count + 1.
                     sum = sum + ftt
                  ENDIF

*  If any data present, set the value at (x,y)
*  If not, bad_flag_result already has been assigned
                  IF (count .GT. 0.) THEN
                     result(i,j,k,l,m,n) = sum/count
                  ENDIF

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
 400        CONTINUE

               m1 = m1 + arg_incr(E_AXIS,ARG1)
 500        CONTINUE

               n1 = n1 + arg_incr(F_AXIS,ARG1)
 600        CONTINUE

         ENDIF  ! IF (ibot .GE. ...  .AND. itop .LE.  ...

* Only one of the ARG4 and one of the ARG5 actually increments
         i4 = i4 + arg_incr(X_AXIS,ARG4)
         j4 = j4 + arg_incr(Y_AXIS,ARG4)
         k4 = k4 + arg_incr(Z_AXIS,ARG4)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         m4 = m4 + arg_incr(E_AXIS,ARG4)
         n4 = n4 + arg_incr(F_AXIS,ARG4)

         i5 = i5 + arg_incr(X_AXIS,ARG5)
         j5 = j5 + arg_incr(Y_AXIS,ARG5)
         k5 = k5 + arg_incr(Z_AXIS,ARG5)
         l5 = l5 + arg_incr(T_AXIS,ARG5)
         m5 = m5 + arg_incr(E_AXIS,ARG5)
         n5 = n5 + arg_incr(F_AXIS,ARG5)
 700  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)

      END
