*           rr_grad_ps_u.F
*
*           Rick Romea
*           Jan. 24, 2000
*
*     Computes the surface pressure gradient term in the u momentum equation
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
************************************************************************

      SUBROUTINE RR_grad_ps_u_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'Surface (barotropic) -dp/dx term (cm/s^2) on the MOM2 U Grid')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_has_vari_args    (id, NO)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, YES, YES, YES, YES)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity,required to access MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,0,+1)
      
      CALL ef_set_arg_name         (id, ARG2, 'SURF_PRESS')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Surface Pressure, on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'g/cm/sec^2')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO, NO, YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,0,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,0,+1)
      END

************************************************************************

      SUBROUTINE RR_grad_ps_u_compute(id, arg_1, arg_2, result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS)
      REAL bad_flag_result
      REAL arg_1  (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .             mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2  (mem2lox:mem2hix, mem2loy:mem2hiy, 
     .             mem2loz:mem2hiz, mem2lot:mem2hit)
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

      REAL*8         xU(1024)
      REAL*8         yU(1024)

      INTEGER        ilat
      INTEGER        ilon

      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      INCLUDE 'rr_parameters.h'

      REAL       csu
      REAL       dxt

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)

      ! Get U grid coordinates

      CALL ef_get_coordinates    (id, ARG1, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG1),
     .                            arg_hi_ss(X_AXIS,ARG1), xU)    
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)

      ! check axis units

      CALL ef_get_axis_info  (id, ARG2, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(1) .NE. 'deg'       .AND.
     .         ax_units(1) .NE. 'lon'       .AND.
     .         ax_units(1) .NE. 'degrees_E' .AND.
     .         ax_units(1) .NE. 'longitude' .AND.
     .         ax_units(1) .NE. 'Longitude' .AND. 
     .         ax_units(1) .NE. 'LONGITUDE' ) THEN
        WRITE (6,*)'Longitude axis units =', ax_units(1) 
        ! CALL ef_bail_out(id,'Longitude axis must be in degrees')
      ENDIF

      IF     ( ax_units(2) .NE. 'deg'       .AND.
     .         ax_units(2) .NE. 'lat'       .AND.
     .         ax_units(2) .NE. 'degrees_N' .AND.
     .         ax_units(2) .NE. 'latitude'  .AND.
     .         ax_units(2) .NE. 'Latitude'  .AND. 
     .         ax_units(2) .NE. 'LATITUDE' ) THEN
        WRITE (6,*)'Latitude axis units =', ax_units(2) 
        ! CALL ef_bail_out(id,'Latitude axis must be in degrees')
      ENDIF

      ! start the loop

      iLat = 1
      j1 = arg_lo_ss(Y_AXIS,ARG2) 
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         csu = COS(SNGL(yU(iLat))*Degrees_to_radians) 

         iLon = 1
         i1 = arg_lo_ss(X_AXIS,ARG2) 
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            dxt = SNGL (xU(iLon+1)-xU(iLon)) * Longitude_to_cm

            k1 = arg_lo_ss(Z_AXIS,ARG2)
            DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG2)
               DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF (
     .              arg_2(i1,  j1,  k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1,  j1+1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1,  j1-1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1+1,j1,  k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1+1,j1+1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1+1,j1-1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1-1,j1,  k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1-1,j1+1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i1-1,j1-1,k1,l1) .EQ. bad_flag(ARG2) 
     .                      ) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) = - (
     .                            arg_2(i1+1,j1+1,k1,l1)
     .                        +   arg_2(i1+1,j1  ,k1,l1)
     .                        -   arg_2(i1  ,j1+1,k1,l1)
     .                        -   arg_2(i1  ,j1  ,k1,l1)
     .                ) / 2. / rho_zero / csu / dxt
     .                 * sec_per_month
                  ENDIF

                  l1 = l1 + arg_incr(T_AXIS,ARG2)
               ENDDO

               k1 = k1 + arg_incr(Z_AXIS,ARG2)
            ENDDO

            iLon = iLon + 1
            i1 = i1 + arg_incr(X_AXIS,ARG2)
         ENDDO

         iLat = iLat + 1
         j1 = j1 + arg_incr(Y_AXIS,ARG2)
      ENDDO
      END











