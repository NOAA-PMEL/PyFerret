*           rr_adv_sz.F
*
*           Rick Romea
*           Jan. 24, 2000      
*
*     Zonal nonlinear advective term:  - w dS/dz
*     Units :  (SALINITY(ppm) - 0.035) / s
*     MOM2 Grid:  T   
*
*******************************************************************

      SUBROUTINE RR_adv_sz_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'-w(dS/dz) nonlinear term (SALT/s);MOM2 T Grid.')
      CALL ef_set_num_args         (id,3)
      CALL ef_set_axis_inheritance (id,IMPLIED_BY_ARGS, 
     .                                 IMPLIED_BY_ARGS, 
     .                                 IMPLIED_BY_ARGS, 
     .                                 IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, NO, NO)
      CALL ef_set_arg_name         (id,ARG1,'U')
      CALL ef_set_arg_desc         (id,ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id,ARG1,'cm/sec')
      CALL ef_set_arg_type         (id,ARG1,FLOAT_ARG)
      CALL ef_set_axis_influence   (id,ARG1,YES,YES,YES,YES)
      CALL ef_set_axis_extend      (id,ARG1,X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG1,Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG1,Z_AXIS,0,+1)

      CALL ef_set_arg_name         (id,ARG2,'W')
      CALL ef_set_arg_desc         (id,ARG2,
     . 'Vertical velocity, at T cell bottom on the MOM2 Grid.')
      CALL ef_set_arg_unit         (id,ARG2,'cm/sec')
      CALL ef_set_arg_type         (id,ARG2,FLOAT_ARG)
      CALL ef_set_axis_influence   (id,ARG2,NO,NO,NO,YES)
      CALL ef_set_axis_extend      (id,ARG2,X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG2,Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG2,Z_AXIS,0,+1)

      CALL ef_set_arg_name         (id,ARG3,'SALT')
      CALL ef_set_arg_desc         (id,ARG3,
     . 'Salinity, on the MOM2 T Grid.')
      CALL ef_set_arg_unit         (id,ARG3,'(ppt-35)/1000')
      CALL ef_set_arg_type         (id,ARG3,FLOAT_ARG)
      CALL ef_set_axis_influence   (id,ARG3, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id,ARG3,X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG3,Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id,ARG3,Z_AXIS,0,+1)
      END




      SUBROUTINE RR_adv_sz_compute(id, arg_1, arg_2, arg_3, result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER        id

      REAL           bad_flag(EF_MAX_ARGS)
      REAL           bad_flag_result
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

      REAL*8         yU(1024)
      REAL*8         xU(1024)
      REAL*8         zW(1024)
      REAL*8         yT(1024)
      REAL*8         xT(1024)

      INTEGER        ilat
      INTEGER        ilon
      INTEGER        iZ

      CHARACTER*16   ax_name(4)
      CHARACTER*16   ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL           adv_fb         

      INCLUDE 'rr_parameters.h'

      adv_fb(i2,j2,k2,l2,i3,j3,k3,l3) =  arg_2(i2,j2,k2,l2)
     .                  * (arg_3(i3,j3,k3,l3) + arg_3(i3,j3,k3+1,l3))

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG1),
     .                            arg_hi_ss(X_AXIS,ARG1), xU)
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)
      CALL ef_get_coordinates    (id, ARG2, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG2),
     .                            arg_hi_ss(Z_AXIS,ARG2), zW)
      CALL ef_get_coordinates    (id, ARG3, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG3),
     .                            arg_hi_ss(X_AXIS,ARG3), xT)
      CALL ef_get_coordinates    (id, ARG3, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG3),
     .                            arg_hi_ss(Y_AXIS,ARG3), yT)
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
     .         ax_units(2) .NE. 'degrees_N' .AND.
     .         ax_units(2) .NE. 'lat'       .AND.
     .         ax_units(2) .NE. 'latitude'  .AND.
     .         ax_units(2) .NE. 'Latitude'  .AND. 
     .         ax_units(2) .NE. 'LATITUDE' ) THEN
        WRITE (6,*)'Latitude axis units =', ax_units(2) 
        !CALL ef_bail_out(id,'Latitude axis must be in degrees')
      ENDIF
      CALL ef_get_axis_info  (id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(3) .NE. 'm'           .AND.
     .         ax_units(3) .NE. 'meters'  .AND.
     .         ax_units(3) .NE. 'M'           .AND.
     .         ax_units(3) .NE. 'METERS' ) THEN
        WRITE (6,*)'Longitude axis units =', ax_units(1) 
        !CALL ef_bail_out(id,'Z axis must be in meters')
      ENDIF 

      l2 = arg_lo_ss(T_AXIS,ARG2) 
      l3 = arg_lo_ss(T_AXIS,ARG3) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         ilat = 2  
         j2 = arg_lo_ss(Y_AXIS,ARG2) + 1
         j3 = arg_lo_ss(Y_AXIS,ARG3) + 1
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            iLon = 2
            i2 = arg_lo_ss(X_AXIS,ARG2) + 1
            i3 = arg_lo_ss(X_AXIS,ARG3) + 1
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
	       iZ = 1
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
                  IF (arg_2(i2,  j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_3(i3,  j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3,  j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3,  j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3-1,k3,l3) .EQ. bad_flag(ARG3)) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     IF (iZ .EQ. 1) THEN
                        result(i,j,k,l) = (  
     .                       arg_2(i2,j2,k2,l2)
     .                  * (arg_3(i3,j3,k3,l3) + arg_3(i3,j3,k3+1,l3))
     .                           / 2. 
     .                       - arg_3(i3,j3,k3,l3) * 
     .                              arg_2(i2,j2,k2,l2)  
     .                    ) / (zW(iZ)*Meters_to_cm)     
     .                     * sec_per_month
                     ELSE
                        result(i,j,k,l) =  ( 
     .                      - (adv_fb(i2,j2,k2-1,l2,i3,j3,k3-1,l3)
     .                        - adv_fb(i2,j2,k2,l2,i3,j3,k3,l3)) / 2.
     .                      + arg_3(i3,j3,k3,l3) * (
     .                          arg_2(i2,j2,k2-1,l2) - 
     .                              arg_2(i2,j2,k2,l2) ) 
     .                     )  /(zW(iZ-1)-zW(iZ))/Meters_to_cm        
     .                     * sec_per_month
                     ENDIF
                  ENDIF
                  iZ = iZ + 1
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
                  k3 = k3 + arg_incr(Z_AXIS,ARG3)
               ENDDO
               iLon = iLon + 1
               i2 = i2 + arg_incr(X_AXIS,ARG2)
               i3 = i3 + arg_incr(X_AXIS,ARG3)
            ENDDO
            iLat = iLat + 1
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
         ENDDO
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
      ENDDO               
      END

