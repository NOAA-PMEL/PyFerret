      SUBROUTINE RR_w_Qiao_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,'w integral from Qiao (1997)')
      CALL ef_set_num_args         (id, 5)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)

      CALL ef_set_arg_name         (id, ARG1, 'u4') 
      CALL ef_set_arg_desc         (id, ARG1, 'u4')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, Z_AXIS,0,+1) 
 
      CALL ef_set_arg_name         (id, ARG2, 'u1')
      CALL ef_set_arg_desc         (id, ARG2, 'u1')
      CALL ef_set_arg_unit         (id, ARG2, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2,  NO, YES, YES, YES)

      CALL ef_set_arg_name         (id, ARG3, 'u3')
      CALL ef_set_arg_desc         (id, ARG3, 'u3')
      CALL ef_set_arg_unit         (id, ARG3, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3,  NO, YES, YES, YES)

      CALL ef_set_arg_name         (id, ARG4, 'v2')
      CALL ef_set_arg_desc         (id, ARG4, 'v2')
      CALL ef_set_arg_unit         (id, ARG4, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG4, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG4, YES,  NO, YES, YES)

      CALL ef_set_arg_name         (id, ARG5, 'v5')
      CALL ef_set_arg_desc         (id, ARG5, 'v5')
      CALL ef_set_arg_unit         (id, ARG5, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG5, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG5, YES,  NO, YES, YES)

      END


      SUBROUTINE RR_w_Qiao_compute(id,arg_1,arg_2,arg_3,arg_4,
     .                             arg_5,result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER        id
      REAL           arg_1 (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .                      mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL           arg_2 (mem2lox:mem2hix, mem2loy:mem2hiy, 
     .                      mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL           arg_3 (mem3lox:mem3hix, mem3loy:mem3hiy, 
     .                      mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL           arg_4 (mem4lox:mem4hix, mem4loy:mem4hiy, 
     .                      mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL           arg_5 (mem5lox:mem5hix, mem5loy:mem5hiy, 
     .                      mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL           result (memreslox:memreshix,memresloy:memreshiy,
     .                       memresloz:memreshiz,memreslot:memreshit)
      INTEGER        res_lo_ss (4)
      INTEGER        res_hi_ss (4)
      INTEGER        res_incr  (4)
      INTEGER        arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER        arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER        arg_incr  (4,EF_MAX_ARGS)

      INTEGER        i,  j,  k,  l
      INTEGER        i5, j5, k5, l5
      INTEGER        i4, j4, k4, l4
      INTEGER        i3, j3, k3, l3
      INTEGER        i2, j2, k2, l2

      REAL*8         zW(1024)
      INTEGER        iZ
      REAL           dx, dy, dz, sum

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG1),
     .                            arg_hi_ss(Z_AXIS,ARG1), zW)

      dx = 425.2e5  ! 4 degrees of longitude in cm (at the equator)
      dy = 212.7e5  ! 2 degrees of latitude in cm

      ! Single index

      i  = res_lo_ss(X_AXIS)
      i2 = arg_lo_ss(X_AXIS,ARG2) 
      i3 = arg_lo_ss(X_AXIS,ARG3) 
      i4 = arg_lo_ss(X_AXIS,ARG4) 
      i5 = arg_lo_ss(X_AXIS,ARG5)

      j  = res_lo_ss(Y_AXIS)
      j2 = arg_lo_ss(Y_AXIS,ARG2) 
      j3 = arg_lo_ss(Y_AXIS,ARG3) 
      j4 = arg_lo_ss(Y_AXIS,ARG4) 
      j5 = arg_lo_ss(Y_AXIS,ARG5)

      l2 = arg_lo_ss(T_AXIS,ARG2) 
      l3 = arg_lo_ss(T_AXIS,ARG3) 
      l4 = arg_lo_ss(T_AXIS,ARG4) 
      l5 = arg_lo_ss(T_AXIS,ARG5) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         iZ = 1
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         k3 = arg_lo_ss(Z_AXIS,ARG3)
         k4 = arg_lo_ss(Z_AXIS,ARG4)
         k5 = arg_lo_ss(Z_AXIS,ARG5)

         sum  = 0.

         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

           IF(k.EQ.1)THEN
              dz = 100. * (zW(1)+zW(2))/2.
           ELSE
              dz = 100. * (zW(iZ+1)-zW(iZ-1))/2.
           ENDIF

           !  ARG_1 = u4 
           !  ARG_2 = u1
           !  ARG_3 = u3
           !  ARG_4 = v2
           !  ARG_5 = v5
 
           sum = sum - (
     .        (ARG_3(i3,j3,k3,l3)-ARG_2(i2,j2,k2,l2))/dx +
     .        (ARG_4(i4,j4,k4,l4)-ARG_5(i5,j5,k5,l5))/dy ) * dz 

           result(i,j,k,l) = sum

           iZ = iZ + 1
           k2 = k2 + arg_incr(Z_AXIS,ARG2)
           k3 = k3 + arg_incr(Z_AXIS,ARG3)
           k4 = k4 + arg_incr(Z_AXIS,ARG4)
           k5 = k5 + arg_incr(Z_AXIS,ARG5)
         ENDDO 

         l5 = l5 + arg_incr(T_AXIS,ARG5)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
      ENDDO               
      END				






