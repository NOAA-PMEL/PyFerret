*           rr_adv_sx.F
*
*           Rick Romea
*           Jan. 24, 2000 
*
*     Computes the zonal nonlinear advective term: - u dS/dx
*     Units :      (SALINITY(ppm) - 0.035) / s
*     MOM2 Grid:  T  
*
*******************************************************************

      SUBROUTINE RR_adv_sx_init(id)
	IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'-u(dS/dx) nonlinear term (SALT/s);MOM2 T Grid.')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, YES, YES)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)

      CALL ef_set_arg_name         (id, ARG2, 'SALT')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Salinity, on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG2,
     . '(SALINITY(ppt) - 35) /1000')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)

      END


      SUBROUTINE RR_adv_sx_compute(id, arg_1, arg_2, result)
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
      INTEGER        i2, j2, k2, l2 

      REAL*8         yU(1024)
      REAL*8         xU(1024)
      REAL*8         yT(1024)
      REAL*8         xT(1024)
      INTEGER        ilat
      INTEGER        ilon
      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL dxt, dyt, dyu
      REAL adv_vet, adv_fe, cst

	INCLUDE 'rr_parameters.h'

C*********************************************************************
c     dyt     = latitudinal height of "t" grid box (in cm)
c     dxt     = longitudinal width of "t" grid box at the equator (cm)
c     dyu     = latitudinal height of "u,v" grid box (cm)
c     yu(j)   = latitude of the jth "u,v" point in degrees 
c     yt(j)   = latitude of the jth "t" point in degrees   
c     xt(i)   = longitude of the ith "t" point in degrees  
c     xu(i)   = longitude of the ith "u,v" point in degrees
C*********************************************************************

      !  Statement functions

      dxt(i) = SNGL (xU(i)   - xU(i-1)) * Longitude_to_cm
      dyt(j) = SNGL (yU(j)   - yU(j-1)) * Latitude_to_cm
      dyu(j) = SNGL (yT(j+1) - yT(j))   * Latitude_to_cm

      adv_vet(i1,j1,k1,l1,iLat) = (arg_1(i1,j1,k1,l1)*dyu(iLat) + 
     .              arg_1(i1,j1-1,k1,l1)*dyu(iLat-1)) /2./ dyt(iLat)
      adv_fe (i1,j1,k1,l1,i2,j2,k2,l2,iLat) = adv_vet(i1,j1,k1,l1,iLat)
     .                  *(arg_2(i2,j2,k2,l2) + arg_2(i2+1,j2,k2,l2))

      ! Get axis data

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG1),
     .                            arg_hi_ss(X_AXIS,ARG1), xU)
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)
      CALL ef_get_coordinates    (id, ARG2, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG2),
     .                            arg_hi_ss(X_AXIS,ARG2), xT)
      CALL ef_get_coordinates    (id, ARG2, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG2),
     .                            arg_hi_ss(Y_AXIS,ARG2), yT)

      ! Check axis units:  bail out if not lat and lon.

      CALL ef_get_axis_info      (id, ARG1, ax_name, ax_units, 
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
      l2 = arg_lo_ss(T_AXIS,ARG2) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         
         k1 = arg_lo_ss(Z_AXIS,ARG1)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            ilat = 2  
            j1 = arg_lo_ss(Y_AXIS,ARG1) + 1
            j2 = arg_lo_ss(Y_AXIS,ARG2) + 1
            DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               cst = cos(SNGL(yT(iLat)) * Degrees_to_radians) 

               iLon = 2
               i1 = arg_lo_ss(X_AXIS,ARG1) + 1
               i2 = arg_lo_ss(X_AXIS,ARG2) + 1
               DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                  IF (
     .                arg_1(i1,  j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,  j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,  j1-1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1-1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1-1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_2(i2,  j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2-1,k2,l2) .EQ. bad_flag(ARG2)
     .                  )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) =  (
     .                  - (adv_fe(i1,j1,k1,l1,i2,j2,k2,l2,iLat) - 
     .                    adv_fe(i1-1,j1,k1,l1,i2-1,j2,k2,l2,iLat)) 
     .                       /2.
     .                  + arg_2(i2,j2,k2,l2) *
     .                     (  adv_vet(i1,j1,k1,l1,iLat) 
     .                     - adv_vet(i1-1,j1,k1,l1,iLat) )   ) 
     .                       /dxt(iLon)/cst*sec_per_month
                  ENDIF
                  iLon = iLon + 1
                  i1 = i1 + arg_incr(X_AXIS,ARG1)
                  i2 = i2 + arg_incr(X_AXIS,ARG2)
               ENDDO

               iLat = iLat + 1
               j1 = j1 + arg_incr(Y_AXIS,ARG1)
               j2 = j2 + arg_incr(Y_AXIS,ARG2)
            ENDDO               

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
            k2 = k2 + arg_incr(Z_AXIS,ARG2)
         ENDDO

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
      ENDDO

      END











