*
* curv_range.F
*
* Ansley Manke
* March 2005
*
* This function finds the indices of the bounding box for taking 
* a subset of a variable defined on a curvilinear grid.
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


      SUBROUTINE curv_range_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER*120 descr

      WRITE (descr, 100)
      CALL ef_set_desc(id, descr )

  100 FORMAT (
     . 'find i,j bounds for subset of a variable in curvilinear ',
     . 'coordinates' )

      CALL ef_set_num_args(id, 7)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, NORMAL, NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'LONGITUDES')
      CALL ef_set_arg_desc(id, arg, 
     .             '2-D longitudes of curvilinear grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'LATITUDES')
      CALL ef_set_arg_desc(id, arg, 
     .             '2-D latitudes of curvilinear grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'xrange_lo')
      CALL ef_set_arg_desc(id, arg, 'Minimum of longitude range')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'xrange_hi')
      CALL ef_set_arg_desc(id, arg, 'Maximum of longitude range')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'yrange_lo')
      CALL ef_set_arg_desc(id, arg, 'Minimum of latitude range')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'yrange_hi')
      CALL ef_set_arg_desc(id, arg, 'Maximum of latitude range')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'modulo flag for X coordinates')
      CALL ef_set_arg_desc(id, arg, '1= X modulo; 0= X not modulo')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


      SUBROUTINE curv_range_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_axis_limits(id, X_AXIS, 1, 4)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE curv_range_compute(id, arg_1, arg_2, 
     .           arg_3, arg_4, arg_5, arg_6, arg_7, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id, arg

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, 
     .           mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, 
     .           mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, 
     .           mem6loz:mem6hiz, mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, 
     .           mem7loz:mem7hiz, mem7lot:mem7hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l
      INTEGER ilo, ihi, jlo, jhi, nrep, irep
      REAL small, xrange_lo, xrange_hi, yrange_lo, yrange_hi, xx, xxp, 
     .     yy, yyp, big, xadd, xmin, xmax, xmin_field, xmax_field, 
     .     ymin_field, ymax_field, xmod, xdiff_lo, xdiff_hi, ydiff_lo, 
     .     ydiff_hi
      LOGICAL xmodulo, found_range
      CHARACTER*100 errmsg

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      arg = 3
      CALL ef_get_one_val(id, arg, xrange_lo)
      arg = 4
      CALL ef_get_one_val(id, arg, xrange_hi)
      arg = 5
      CALL ef_get_one_val(id, arg, yrange_lo)
      arg = 6
      CALL ef_get_one_val(id, arg, yrange_hi)
      arg = 7
      CALL ef_get_one_val(id, arg, xmod)
      xmodulo = (xmod .GT. 0.)

c curv_range_test
c* Set the values to some particular ones, to see if this is the function being executed.
c
c      j = res_lo_ss(Y_AXIS)
c      k = res_lo_ss(Z_AXIS)
c      l = res_lo_ss(T_AXIS)
c
c      result(1,j,k,l) = 11
c      result(2,j,k,l) = 22
c      result(3,j,k,l) = 33
c      result(4,j,k,l) = 44
c      RETURN

* See if the user asked for the whole range of arg_1 and arg_2 


      big = MAX (1.e34, ABS(bad_flag(ARG1)) )
      big = MAX (big, ABS(bad_flag(ARG2)) )
      xmin_field = big
      xmax_field = -1.* big
      ymin_field = big
      ymax_field = -1.* big
      DO 53 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
      DO 52 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 51 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 50 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
      
         xmin_field = MIN(xmin_field, arg_1(i,j,k,l))
         xmax_field = MAX(xmax_field, arg_1(i,j,k,l))
         ymin_field = MIN(ymin_field, arg_2(i,j,k,l))
         ymax_field = MAX(ymax_field, arg_2(i,j,k,l))

 50   CONTINUE
 51   CONTINUE
 52   CONTINUE
 53   CONTINUE

      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

! use the range of the data field to decide how much precision 
! to require for **requested range equals the data range**

      small = 0.01* (xmax_field - xmin_field)
      small = MIN(small, 0.01* (ymax_field - ymin_field) )

      IF (small .LE. 0) THEN  ! if x or y range is 0 use the other direction
         small = 0.01* (xmax_field - xmin_field)
         small = MAX(small, 0.01* (ymax_field - ymin_field) )
      ENDIF

      IF (small .LE. 0) small = 1.e-4

      xdiff_lo = ABS(xmin_field - xrange_lo) 
      xdiff_hi = ABS(xmax_field - xrange_hi) 
      ydiff_lo = ABS(ymin_field - yrange_lo) 
      ydiff_hi = ABS(ymax_field - yrange_hi) 

      IF (xdiff_lo .LT. small .AND. ydiff_lo .LT. small  .AND.
     .    xdiff_hi .LT. small .AND. ydiff_hi .LT. small) THEN 
         result(1,j,k,l) = arg_lo_ss(X_AXIS, ARG1)
         result(2,j,k,l) = arg_hi_ss(X_AXIS, ARG1)
         result(3,j,k,l) = arg_lo_ss(Y_AXIS, ARG2)
         result(4,j,k,l) = arg_hi_ss(Y_AXIS, ARG2)
         RETURN
      ENDIF

      IF (ydiff_lo .LT. small .AND. ydiff_hi .LT. small .AND. 
     .     xmodulo) THEN
         nrep = NINT((xrange_lo-xmin_field)/360.)
         xdiff_lo = ABS(xmin_field+360.*nrep - xrange_lo)
         nrep = NINT((xrange_hi-xmax_field)/360.)
         xdiff_hi = ABS(xmax_field+360.*nrep - xrange_hi)
         IF (xdiff_lo .LT. small .AND. xdiff_hi .LT. small ) THEN
            result(1,j,k,l) = arg_lo_ss(X_AXIS, ARG1)
            result(2,j,k,l) = arg_hi_ss(X_AXIS, ARG1)
            result(3,j,k,l) = arg_lo_ss(Y_AXIS, ARG2)
            result(4,j,k,l) = arg_hi_ss(Y_AXIS, ARG2)
            RETURN
         ENDIF
      ENDIF

* If the requested range is larger than the data, return whole range

      IF (.NOT. xmodulo) THEN  
         IF (xrange_lo.LT.xmin_field .AND. xrange_hi.GT.xmax_field .AND.
     .       yrange_lo.LT.ymin_field .AND. yrange_hi.GT.ymax_field)THEN
            result(1,j,k,l) = arg_lo_ss(X_AXIS, ARG1)
            result(2,j,k,l) = arg_hi_ss(X_AXIS, ARG1)
            result(3,j,k,l) = arg_lo_ss(Y_AXIS, ARG2)
            result(4,j,k,l) = arg_hi_ss(Y_AXIS, ARG2)
            RETURN
         ENDIF
      ELSE

* If x is to be treated as modulo, use the code below to determine the indices, but
* if the Y range is larger than the data, use the data min and/or max for y range.
         IF (yrange_lo .LT. ymin_field) yrange_lo = ymin_field
         IF (yrange_hi .GT. ymax_field) yrange_hi = ymax_field
      ENDIF



* Find range of I,J for which arg_1 is in xrange_lo, xrange_hi
* and arg_2 is in yrange_lo, yrange_hi

      ilo = arg_hi_ss(X_AXIS, ARG1)
      ihi = arg_lo_ss(X_AXIS, ARG1)
      jlo = arg_hi_ss(Y_AXIS, ARG2)
      jhi = arg_lo_ss(y_AXIS, ARG2)

      big = ABS(bad_flag(ARG1) )
      xmin = big
      xmax = -1.* big
      found_range = .FALSE.

      DO 400 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
      DO 300 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 200 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 100 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

         xx = arg_1(i,j,k,l)
         yy = arg_2(i,j,k,l)
         IF ( xx .EQ. bad_flag(1) ) GOTO 5500
         IF ( yy .EQ. bad_flag(2) ) GOTO 5500

         IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi .AND.
     .        yy .GE. yrange_lo  .AND. yy .LE. yrange_hi ) THEN
            found_range = .TRUE.
            ilo = MIN(i,ilo)
            ihi = MAX(i,ihi)
            jlo = MIN(j,jlo)
            jhi = MAX(j,jhi)
            xmin = MIN(xmin, xx)
            xmax = MAX(xmin, xx)
         ENDIF

 100  CONTINUE
 200  CONTINUE
 300  CONTINUE
 400  CONTINUE

* If no range found, is the range strictly between two adjacent points
* in either direction?

      IF (.NOT. found_range .AND. 
     .    xrange_lo.GE.xmin_field .AND. xrange_hi.LE.xmax_field .AND.
     .    yrange_lo.GE.ymin_field .AND. yrange_hi.LE.ymax_field ) THEN
     
         ilo = arg_hi_ss(X_AXIS, ARG1)
         ihi = arg_lo_ss(X_AXIS, ARG1)
         jlo = arg_hi_ss(Y_AXIS, ARG2)
         jhi = arg_lo_ss(y_AXIS, ARG2)

         DO 401 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)-1
         DO 301 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)-1
         DO 201 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
         DO 101 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

            xx = arg_1(i,j,k,l)
            xxp = arg_1(i+1,j,k,l)
            yy = arg_2(i,j,k,l)
            yyp = arg_2(i,j+1,k,l)

            IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi .AND.
     .           yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
               found_range = .TRUE.
               ilo = min(i, ilo)
               ihi = MAX(i, ihi)
               jlo = min(j, jlo)
               jhi = MAX(j+1, jhi)
            ENDIF

            IF ( yy .GE. yrange_lo  .AND. yy .LE. yrange_hi .AND.
     .           xrange_lo .GE. xx .AND. xrange_hi .LT. xxp) THEN
               found_range = .TRUE.
               ilo = min(i, ilo)
               ihi = MAX(i+1, ihi)
               jlo = min(j, jlo)
               jhi = MAX(j, jhi)
            ENDIF

            IF ( xrange_lo .GE. xx .AND. xrange_hi .LT. xxp .AND.
     .           yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
               found_range = .TRUE.
               ilo = min(i, ilo)
               ihi = MAX(i+1, ihi)
               jlo = min(j, jlo)
               jhi = MAX(j+1, jhi)
            ENDIF

 101     CONTINUE
 201     CONTINUE
 301     CONTINUE
 401     CONTINUE
      ENDIF

      IF (xmodulo) THEN
         IF (xmin .EQ. big) xmin = xmin_field
         IF (xmax .EQ. -1.*big) xmax = xmax_field
 
* Does the range found cover the requested xrange_lo,xrange_hi range?
* If not we will need some modulo repetitions.

         IF (xmin .GT. xrange_lo) THEN
            nrep = NINT((xmin-xrange_lo)/360)
            xadd = 0.
            DO 501 irep = 1, nrep
               xadd = xadd - 360.

* Previous loop found no overlap of x coordinates and desired output range.
* start over with xmax, xmin find whole range

               IF (xmin .GT. xrange_hi) THEN
                  xmin =  big 
                  xmax = -1. * big   
               ENDIF
     
               ilo = arg_hi_ss(X_AXIS, ARG1)
               ihi = arg_lo_ss(X_AXIS, ARG1)
               jlo = arg_hi_ss(Y_AXIS, ARG2)
               jhi = arg_lo_ss(y_AXIS, ARG2)

               DO 402 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
               DO 302 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
               DO 202 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
               DO 102 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

                  xx = arg_1(i,j,k,l) + xadd
                  yy = arg_2(i,j,k,l)

                  IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi  .AND.
     .                 yy .GE. yrange_lo  .AND. yy .GE. yrange_hi ) THEN
                     found_range = .TRUE.
                     ilo = MIN(i,ilo)
                     ihi = MAX(i,ihi) 
                     jlo = MIN(j,jlo)
                     jhi = MAX(j,jhi) 
                     xmin = MIN(xmin, xx)
                     xmax = MAX(xmin, xx)
                  ENDIF

 102           CONTINUE
 202           CONTINUE
 302           CONTINUE
 402           CONTINUE

               IF (.NOT. found_range) THEN
               
                  ilo = arg_hi_ss(X_AXIS, ARG1)
                  ihi = arg_lo_ss(X_AXIS, ARG1)
                  jlo = arg_hi_ss(Y_AXIS, ARG2)
                  jhi = arg_lo_ss(y_AXIS, ARG2)

                  DO 403 i = arg_lo_ss(X_AXIS,ARG1), 
     .               arg_hi_ss(X_AXIS,ARG1)-1
                  DO 303 j = arg_lo_ss(Y_AXIS,ARG1), 
     .               arg_hi_ss(Y_AXIS,ARG1)
                  DO 203 k = arg_lo_ss(Z_AXIS,ARG1), 
     .               arg_hi_ss(Z_AXIS,ARG1)
                  DO 103 l = arg_lo_ss(T_AXIS,ARG1), 
     .               arg_hi_ss(T_AXIS,ARG1)

                  xx = arg_1(i,j,k,l) + xadd
                  xxp = arg_1(i+1,j,k,l) + xadd
                  yy = arg_2(i,j,k,l)

                  IF ( xrange_lo .GE. xx .AND. xrange_hi .LT. xxp  .AND.
     .                 yy .GE. yrange_lo  .AND. yy .GE. yrange_hi ) THEN
                     found_range = .TRUE.
                     ilo = MIN(i,ilo)
                     ihi = MAX(i+1,ihi) 
                     jlo = MIN(j,jlo)
                     jhi = MAX(j,jhi) 
                     xmin = MIN(xmin, xx)
                     xmax = MAX(xmin, xxp)
                  ENDIF

 103              CONTINUE
 203              CONTINUE
 303              CONTINUE
 403              CONTINUE
               ENDIF

               IF (xmin .EQ. big) xmin = xmin_field + xadd
               IF (xmax .EQ. -1.*big) xmax = xmax_field + xadd

 501        CONTINUE

* If no range found, is the range strictly between two adjacent points
* in either direction?

            IF (.NOT. found_range .AND. 
     .     xrange_lo.GE.xmin_field .AND. xrange_hi.LE.xmax_field .AND.
     .     yrange_lo.GE.ymin_field .AND. yrange_hi.LE.ymax_field ) THEN
                       
               ilo = arg_hi_ss(X_AXIS, ARG1)
               ihi = arg_lo_ss(X_AXIS, ARG1)
               jlo = arg_hi_ss(Y_AXIS, ARG2)
               jhi = arg_lo_ss(y_AXIS, ARG2)

               DO 404 i =arg_lo_ss(X_AXIS,ARG1),arg_hi_ss(X_AXIS,ARG1)-1
               DO 304 j =arg_lo_ss(Y_AXIS,ARG1),arg_hi_ss(Y_AXIS,ARG1)-1
               DO 204 k =arg_lo_ss(Z_AXIS,ARG1),arg_hi_ss(Z_AXIS,ARG1)
               DO 104 l =arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)

                  xx = arg_1(i,j,k,l)+xadd
                  xxp = arg_1(i+1,j,k,l)+xadd
                  yy = arg_2(i,j,k,l)
                  yyp = arg_2(i,j+1,k,l)
      
                  IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi .AND.
     .              yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i, ihi)
                     jlo = min(j, jlo)
                     jhi = MAX(j+1, jhi)
                  ENDIF

                  IF ( yy .GE. yrange_lo  .AND. yy .LE. yrange_hi .AND.
     .                 xrange_lo .GE. xx .AND. xrange_hi .LT. xxp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                     jlo = min(j, jlo)
                     jhi = MAX(j, jhi)
                  ENDIF

                  IF ( xrange_lo .GE. xx .AND. xrange_hi .LT. xxp .AND.
     .                 yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                  ENDIF

 104           CONTINUE
 204           CONTINUE
 304           CONTINUE
 404           CONTINUE
            ENDIF

         ENDIF
   
         IF (xmax .LT. xrange_hi) THEN
            nrep = NINT((xrange_hi-xmax)/360.)
            xadd = 0.
               DO 502 irep = 1, nrep
               xadd = xadd + 360.

* Previous loop found no overlap of x coordinates and desired output range.
* start over with xmax, xmin find whole range

               IF (xmax .LT. xrange_lo) THEN
                  xmin =  big 
                  xmax = -1. * big   
               ENDIF

               ilo = arg_hi_ss(X_AXIS, ARG1)
               ihi = arg_lo_ss(X_AXIS, ARG1)
               jlo = arg_hi_ss(Y_AXIS, ARG2)
               jhi = arg_lo_ss(y_AXIS, ARG2)

               DO 405 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
               DO 305 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
               DO 205 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
               DO 105 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

                  xx = arg_1(i,j,k,l) + xadd
                  yy = arg_2(i,j,k,l)

                  IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi  .AND.
     .                 yy .GE. yrange_lo  .AND. yy .GE. yrange_hi ) THEN
                     found_range = .TRUE.
                     ilo = MIN(i,ilo)
                     ihi = MAX(i,ihi) 
                     jlo = MIN(j,jlo)
                     jhi = MAX(j,jhi) 
                     xmin = MIN(xmin, xx)
                     xmax = MAX(xmin, xx)
                  ENDIF

 105           CONTINUE
 205           CONTINUE
 305           CONTINUE
 405           CONTINUE
 
               IF (.NOT. found_range) THEN
               
                  ilo = arg_hi_ss(X_AXIS, ARG1)
                  ihi = arg_lo_ss(X_AXIS, ARG1)
                  jlo = arg_hi_ss(Y_AXIS, ARG2)
                  jhi = arg_lo_ss(y_AXIS, ARG2)

                  DO 406 i = arg_lo_ss(X_AXIS,ARG1), 
     .               arg_hi_ss(X_AXIS,ARG1)-1
                  DO 306 j = arg_lo_ss(Y_AXIS,ARG1), 
     .               arg_hi_ss(Y_AXIS,ARG1)
                  DO 206 k = arg_lo_ss(Z_AXIS,ARG1), 
     .               arg_hi_ss(Z_AXIS,ARG1)
                  DO 106 l = arg_lo_ss(T_AXIS,ARG1), 
     .               arg_hi_ss(T_AXIS,ARG1)

                     xx = arg_1(i,j,k,l) + xadd
                     xxp = arg_1(i+1,j,k,l) + xadd
                     yy = arg_2(i,j,k,l)
                     yyp = arg_2(i,j+1,k,l)

                     IF ( xrange_lo.GE.xx .AND. xrange_hi.LT.xxp.AND.
     .                    yrange_lo.GE.yy .AND. yrange_hi.LT.yyp ) THEN
                        found_range = .TRUE.
                        ilo = MIN(i,ilo)
                        ihi = MAX(i+1,ihi) 
                        jlo = MIN(j,jlo)
                        jhi = MAX(j,jhi) 
                        xmin = MIN(xmin, xx)
                        xmax = MAX(xmin, xxp)
                     ENDIF

 106              CONTINUE
 206              CONTINUE
 306              CONTINUE
 406              CONTINUE
               ENDIF
 
               IF (xmin .EQ. big) xmin = xmin_field + xadd
               IF (xmax .EQ. -1.*big) xmax = xmax_field + xadd

 502        CONTINUE


* If no range found, is the range strictly between two adjacent points
* in either direction?

            IF (.NOT. found_range .AND.
     .    xrange_lo.GE.xmin_field .AND. xrange_hi.LE.xmax_field .AND.
     .    yrange_lo.GE.ymin_field .AND. yrange_hi.LE.ymax_field ) THEN
     
               ilo = arg_hi_ss(X_AXIS, ARG1)
               ihi = arg_lo_ss(X_AXIS, ARG1)
               jlo = arg_hi_ss(Y_AXIS, ARG2)
               jhi = arg_lo_ss(y_AXIS, ARG2)

               DO 407 i =arg_lo_ss(X_AXIS,ARG1),arg_hi_ss(X_AXIS,ARG1)-1
               DO 307 j =arg_lo_ss(Y_AXIS,ARG1),arg_hi_ss(Y_AXIS,ARG1)-1
               DO 207 k =arg_lo_ss(Z_AXIS,ARG1),arg_hi_ss(Z_AXIS,ARG1)
               DO 107 l =arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)

                  xx = arg_1(i,j,k,l)+xadd
                  xxp = arg_1(i+1,j,k,l)+xadd
                  yy = arg_2(i,j,k,l)
                  yyp = arg_2(i,j+1,k,l)
      
                  IF ( xx .GE. xrange_lo  .AND. xx .LE. xrange_hi .AND.
     .              yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i, ihi)
                     jlo = min(j, jlo)
                     jhi = MAX(j+1, jhi)
                  ENDIF

                  IF ( yy .GE. yrange_lo  .AND. yy .LE. yrange_hi .AND.
     .                 xrange_lo .GE. xx .AND. xrange_hi .LT. xxp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                     jlo = min(j, jlo)
                     jhi = MAX(j, jhi)
                  ENDIF

                  IF ( xrange_lo .GE. xx .AND. xrange_hi .LT. xxp .AND.
     .                 yrange_lo .GE. yy .AND. yrange_hi .LT. yyp) THEN
                     found_range = .TRUE.
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                     ilo = min(i, ilo)
                     ihi = MAX(i+1, ihi)
                  ENDIF

 107           CONTINUE
 207           CONTINUE
 307           CONTINUE
 407           CONTINUE
            ENDIF


         ENDIF

      ENDIF  ! xmodulo


* Return two cells out from the computed result, one to be sure we cover
* the entire requested range (the coords may be cell centers or bounds, and
* a second because the mapping in curv_to_rect uses neighbors around each
* location to compute the result.

      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

      DO 600 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         result(i,j,k,l) = bad_flag_result
 600  CONTINUE

      IF (found_range) THEN
         result(1,j,k,l) = MAX(ilo-2, arg_lo_ss(X_AXIS, ARG1) )
         result(2,j,k,l) = MIN(ihi+2, arg_hi_ss(X_AXIS, ARG1) )
      ENDIF

      IF (found_range) THEN
         result(3,j,k,l) = MAX(jlo-2, arg_lo_ss(Y_AXIS, ARG2) )
         result(4,j,k,l) = MIN(jhi+2, arg_hi_ss(Y_AXIS, ARG2) )
      ENDIF

      RETURN 

 5500 WRITE (errmsg, 1000) 'X', i, j
 1000 FORMAT (A1, 
     . ' coordinates missing where valid value needed, at I,J=', 2i5)

      CALL EF_BAIL_OUT(id, errmsg)

 5600 WRITE (errmsg, 1000) 'Y', i, j
      CALL EF_BAIL_OUT(id, errmsg)


      RETURN
      END
