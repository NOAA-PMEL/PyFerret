*
* list_good.F
*
* Ansley Manke
* April 1, 2005
*
* This function lists the good values of a variable with its location.
*

      SUBROUTINE list_good_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg
      CHARACTER*100 descrip

      WRITE (descrip, 1000)
      CALL ef_set_desc(id, descrip)

 1000 FORMAT(
     .  'List the value and location of valid data. ',
     .  'Return number of valid values')

      CALL ef_set_num_args(id, 1)
      CALL ef_set_num_work_arrays(id, 4)

      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, NORMAL, NORMAL)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Variable')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

      SUBROUTINE list_good_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
  
      call ef_set_axis_limits(id, X_AXIS, 1, 1)

      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE list_good_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER m1, m2

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the dimension of the input arguments for work arrays
*  to hold coordinate values which will be REAL*8

*  xaxsrc

      iwork = 1

      m1 = 1
      m2 = 2*(1 + ABS(arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)))
      IF (arg_lo_ss(X_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
         m1 = 1
         m2 = 1
      ENDIF

      CALL ef_set_work_array_dims (id, iwork, m1,1,1,1, m2,1,1,1)

*  yaxsrc

      iwork = 2
      
      m1 = 1
      m2 = 2*(1 + ABS(arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)))
      IF (arg_lo_ss(Y_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
         m1 = 1
         m2 = 1
      ENDIF

      CALL ef_set_work_array_dims (id, iwork, 1,m1,1,1, 1,m2,1,1)

*  zaxsrc

      iwork = 3

      m1 = 1
      m2 = 2*(1 + ABS(arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1)))
      IF (arg_lo_ss(Z_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
         m1 = 1
         m2 = 1
      ENDIF

      CALL ef_set_work_array_dims (id, iwork, 1,1,m1,1, 1,1,m2,1)

*  taxsrc

      iwork = 4

      m1 = 1
      m2 = 2*(1 + ABS(arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)))
      IF (arg_lo_ss(t_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
         m1 = 1
         m2 = 1
      ENDIF

      CALL ef_set_work_array_dims (id, iwork, 1,1,1,m1, 1,1,1,m2)

      RETURN 
      END
      
*
* In this subroutine we compute the result
*
      SUBROUTINE list_good_compute(id, arg_1, result, 
     .                xaxsrc, yaxsrc, zaxsrc, taxsrc)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

	  INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL*8 xaxsrc(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yaxsrc(wrk2lox:wrk2hix, wrk2loy:wrk2hiy/2,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 zaxsrc(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz/2, wrk3lot:wrk3hit)
      REAL*8 taxsrc(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit/2)
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

      INTEGER i,j,k,l
      INTEGER i1,j1,k1,l1
      REAL xx, yy, zz, tt, count

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG1, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxsrc)

      CALL ef_get_coordinates(id, ARG1, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxsrc)

      CALL ef_get_coordinates(id, ARG1, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG1), arg_hi_ss(Z_AXIS, ARG1), zaxsrc)

      CALL ef_get_coordinates(id, ARG1, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxsrc)

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

c1000  FORMAT (4x, 4i10/, 4x, 4g15.8/  g15.8)
1000  FORMAT (4i10, 4x,  g15.8)

      print *, ' '
      print *, ' good data from variable'
      print *, '       I         J         K         L          VALUE'   
c      print *, '         X         Y         Z         T'
c      print *, ' VALUE'
      print *, ' '

      count = 0
      DO 400 l1 = arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)
      DO 300 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)         

         IF (arg_1(i1,j1,k1,l1) .NE. bad_flag(ARG1)) THEN
           xx = ef_unspecified_int4
           IF (i1 .NE. ef_unspecified_int4) xx =  xaxsrc(i1,1,1,1)
           yy = ef_unspecified_int4
           IF (j1 .NE. ef_unspecified_int4) yy =  yaxsrc(1,j1,1,1)
           zz = ef_unspecified_int4
           IF (k1 .NE. ef_unspecified_int4) zz =  zaxsrc(1,1,k1,1)
           tt = ef_unspecified_int4
           IF (L1 .NE. ef_unspecified_int4) tt =  taxsrc(1,1,1,L1)
           PRINT 1000, i1, j1, k1, l1, 
c     .                 xx, yy, zz, tt,
     .                 arg_1(i1,j1,k1,l1)
           count = count + 1
         ENDIF

 100  CONTINUE
 200  CONTINUE
 300  CONTINUE
 400  CONTINUE

      result(i,j,k,l) = count

      print *, ' '
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
