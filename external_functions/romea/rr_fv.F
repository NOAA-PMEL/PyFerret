*           rr_fv.F
*
*           Rick Romea
*           Jan. 5, 2000 
*
*     Coriolis term in the u momentum equation
*     Equation :  fv
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
************************************************************************

      SUBROUTINE RR_fv_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id, arg
      arg = 1

      CALL ef_set_desc             (id,
     .  'Coriolis term fv (cm/s^2) on the MOM2 U Grid')
      CALL ef_set_num_args         (id, 1)
      CALL ef_set_has_vari_args    (id, NO)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, YES, YES, YES, YES)
      CALL ef_set_arg_name         (id, arg, 'V')
      CALL ef_set_arg_desc         (id, arg,
     . 'Meridional velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, arg, 'cm/sec')
      CALL ef_set_arg_type         (id, arg, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, arg, YES, YES, YES, YES)

      END

************************************************************************

      SUBROUTINE RR_fv_compute(id, arg_1, result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS)
      REAL bad_flag_result
      REAL arg_1  (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .             mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result (memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit)
      INTEGER  res_lo_ss (4)
      INTEGER  res_hi_ss (4)
      INTEGER  res_incr  (4)

      INTEGER  arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER  arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER  arg_incr  (4,EF_MAX_ARGS)

      INTEGER i,  j,  k,  l
      INTEGER i1, j1, k1, l1

      REAL*8     degrees_y(1024)
      INTEGER    iLat
      REAL       Coriolis

      INCLUDE 'rr_parameters.h'

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), degrees_y)

      iLat = 1
      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         Coriolis = Two_Omega*
     .                SIN(SNGL(degrees_y(iLat))*Degrees_to_radians) 

         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1)) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) =  arg_1(i1,j1,k1,l1)*Coriolis
     .                  * sec_per_month
                  ENDIF
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
               ENDDO

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
            ENDDO

            i1 = i1 + arg_incr(X_AXIS,ARG1)
         ENDDO

         iLat = iLat + 1
         j1 = j1 + arg_incr(Y_AXIS,ARG1)
      ENDDO
      END


