      SUBROUTINE RR_adv_tz_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id
      CALL ef_set_desc             (id,
     .'-w(dT/dz) nonlinear term (oC/s);MOM2 T Grid')
      CALL ef_set_num_args         (id, 3)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, NO, NO)
      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Z_AXIS,0,+1)
      CALL ef_set_arg_name         (id, ARG2, 'W')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Vertical velocity, at T cell bottom on the MOM2 Grid. ')
	CALL ef_set_arg_unit         (id, ARG2, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO, NO, YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Z_AXIS,0,+1)
      CALL ef_set_arg_name         (id, ARG3, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG3,
     . 'Potential temperature on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG3, 'deg-C')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, NO, NO, YES, YES)
      CALL ef_set_axis_extend      (id, ARG3, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Z_AXIS,0,+1)
      END


      SUBROUTINE RR_adv_tz_compute(id, arg_1, arg_2, arg_3, result)
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
      REAL           result (memreslox:memreshix,memresloy:memreshiy,
     .                       memresloz:memreshiz,memreslot:memreshit)
      INTEGER        res_lo_ss (4)
      INTEGER        res_hi_ss (4)
      INTEGER        res_incr  (4)
      INTEGER        arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER        arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER        arg_incr  (4,EF_MAX_ARGS)
      INTEGER        i,  j,  k,  l
      INTEGER        i2, j2, k2, l2
      INTEGER        i3, j3, k3, l3
      REAL*8         zW(1024)
      INTEGER        iZ
      REAL           adv_fb         
      INCLUDE 'rr_parameters.h'

      adv_fb(i2,j2,k2,l2,i3,j3,k3,l3) =  arg_2(i2,j2,k2,l2)
     .                  * (arg_3(i3,j3,k3,l3) + arg_3(i3,j3,k3+1,l3))

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_coordinates    (id, ARG2, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG2),
     .                            arg_hi_ss(Z_AXIS,ARG2), zW)
      l2 = arg_lo_ss(T_AXIS,ARG2) 
      l3 = arg_lo_ss(T_AXIS,ARG3) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         j2 = arg_lo_ss(Y_AXIS,ARG2) + 1
         j3 = arg_lo_ss(Y_AXIS,ARG3) + 1
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            i2 = arg_lo_ss(X_AXIS,ARG2) + 1
            i3 = arg_lo_ss(X_AXIS,ARG3) + 1
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               iZ = 1
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
                  IF (iZ .EQ. 1) THEN
                        result(i,j,k,l) =
     .                              arg_2(i2,j2,k2,l2)  
     .                  * (arg_3(i3,j3,k3,l3) - arg_3(i3,j3,k3+1,l3))
     .                                /zW(iZ)/Meters_to_cm 
                  ELSE 
                        result(i,j,k,l) = - ( 
     .      	                adv_fb(i2,j2,k2-1,l2,i3,j3,k3-1,l3)
     .                       - adv_fb(i2,j2,k2,  l2,i3,j3,k3,  l3))
     .                         /2./(zW(iZ-1)-zW(iZ))/Meters_to_cm 
     .                        + arg_3(i3,j3,k3,l3) * ( 
     .                            arg_2(i2,j2,k2-1,l2) - 
     .                               arg_2(i2,j2,k2,l2)) 
     .                             /(zW(iZ-1)-zW(iZ))/Meters_to_cm 
                  ENDIF
                  iZ = iZ + 1
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
                  k3 = k3 + arg_incr(Z_AXIS,ARG3)
               ENDDO
               i2 = i2 + arg_incr(X_AXIS,ARG2)
               i3 = i3 + arg_incr(X_AXIS,ARG3)
            ENDDO
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
         ENDDO
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
      ENDDO               
      END


