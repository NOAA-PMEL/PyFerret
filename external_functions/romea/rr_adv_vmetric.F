*           rr_adv_vmetric.F
*
*           Rick Romea
*           Jan. 5, 2000 
*
*     Meridional nonlinear advective metric: -vv tan(theta) / a
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
*******************************************************************

      SUBROUTINE RR_adv_vmetric_init(id)
	IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id
      CALL ef_set_desc             (id,
     .'Meridional momentum nonlinear metric (cm/s^2);MOM2 U Grid')
      CALL ef_set_num_args         (id, 1)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, YES, YES)
      CALL ef_set_arg_name         (id, ARG1, 'V')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Meridional velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      END


      SUBROUTINE RR_adv_vmetric_compute(id, arg_1, result)
	IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER        id

      REAL           bad_flag(EF_MAX_ARGS)
      REAL           bad_flag_result

      REAL           arg_1 (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .                      mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL           result (memreslox:memreshix,memresloy:memreshiy,
     .                       memresloz:memreshiz,memreslot:memreshit)

      INTEGER        res_lo_ss (4)
      INTEGER        res_hi_ss (4)
      INTEGER        res_incr  (4)

      INTEGER        arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER        arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER        arg_incr  (4,EF_MAX_ARGS)

      INTEGER        i,  j,  k,  l
      INTEGER        i1, j1, k1, l1

      INTEGER        ilat
      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL tng

      REAL*8         yU(1024)

	INCLUDE 'rr_parameters.h'

      ! Get axis data

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)

      ! Check axis units:  bail out if not lat and lon.

      CALL ef_get_axis_info  (id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(1) .NE. 'deg'       .AND.
     .         ax_units(1) .NE. 'lon'       .AND.
     .         ax_units(1) .NE. 'degrees_E' .AND.
     .         ax_units(1) .NE. 'longitude' .AND.
     .         ax_units(1) .NE. 'Longitude' .AND. 
     .         ax_units(1) .NE. 'LONGITUDE' ) THEN
        WRITE (6,*)'Longitude axis units =', ax_units(1) 
        !CALL ef_bail_out(id,'Longitude axis must be in degrees')
      ENDIF

      IF     ( ax_units(2) .NE. 'deg'       .AND.
     .         ax_units(2) .NE. 'lat'       .AND.
     .         ax_units(2) .NE. 'degrees_N' .AND.
     .         ax_units(2) .NE. 'latitude'  .AND.
     .         ax_units(2) .NE. 'Latitude'  .AND. 
     .         ax_units(2) .NE. 'LATITUDE' ) THEN
        WRITE (6,*)'Latitude axis units =', ax_units(2) 
        !CALL ef_bail_out(id,'Latitude axis must be in degrees')
      ENDIF

      l1 = arg_lo_ss(T_AXIS,ARG1) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         
         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            iLat = 1  
            j1 = arg_lo_ss(Y_AXIS,ARG1)
            DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               
               IF ( DABS(yU(iLat)) .EQ. 90.d0 ) THEN
                 tng = 0.   ! handle special case
               ELSE   
                 tng = TAN (SNGL(yU(iLat)) * Degrees_to_radians) 
               ENDIF

               i1 = arg_lo_ss(X_AXIS,ARG1)
               DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1))THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) = - tng / RR_radius
     .                   * ARG_1(i1,j1,k1,l1) 
     .                   * ARG_1(i1,j1,k1,l1) * sec_per_month
 
                  ENDIF
                  i1 = i1 + arg_incr(X_AXIS,ARG1)
               ENDDO

               iLat = iLat + 1
               j1 = j1 + arg_incr(Y_AXIS,ARG1)
            ENDDO               

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
         ENDDO

         l1 = l1 + arg_incr(T_AXIS,ARG1)
      ENDDO

      END











