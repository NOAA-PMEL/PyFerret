*           rr_flux_ux.F
*
*           Rick Romea
*  	    Jan. 24, 2000
*
*     Zonal nonlinear advective term in flux form:  - d(uu)/dx
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
*******************************************************************

      SUBROUTINE RR_flux_ux_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'-d(uu)/dx advective momentum: flux-form (cm/s^2);MOM2 U Grid')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, YES, YES)

      ! Define first argument (U)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)

      !  Extend the x and y axis one point backward and forward

      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)

      ! Define second argument (TEMP)

      CALL ef_set_arg_name         (id, ARG2, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Potential temperature, required to access MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'deg-C')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO, YES, YES)

      !  Extend the x and y axis one point backward and forward

      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)

      END


      SUBROUTINE RR_flux_ux_compute(id, arg_1, arg_2, result)
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
      REAL*8         yU(1024)
      REAL*8         xU(1024)
      REAL*8         yT(1024)
      REAL*8         xT(1024)
      INTEGER        ilat
      INTEGER        ilon
      REAL dxt, dyt, dxu, dyu, dus, dun, duw, due
      REAL adv_vet, adv_veu, adv_fe, csu
      INCLUDE 'rr_parameters.h'

      dxt(i) = SNGL (xU(i)   - xU(i-1)) * Longitude_to_cm
      dyt(j) = SNGL (yU(j)   - yU(j-1)) * Latitude_to_cm
      dxu(i) = SNGL (xT(i+1) - xT(i))   * Longitude_to_cm
      dyu(j) = SNGL (yT(j+1) - yT(j))   * Latitude_to_cm
      dus(j) = SNGL (yU(j)   - yT(j))   * Latitude_to_cm
      dun(j) = SNGL (yT(j+1) - yU(j))   * Latitude_to_cm
      duw(i) = SNGL (xU(i)   - xT(i))   * Longitude_to_cm
      due(i) = SNGL (xT(i+1) - xU(i))   * Longitude_to_cm

      adv_vet(i,j,k,l,iLat) = (arg_1(i,j,k,l)*dyu(iLat) + 
     .              arg_1(i,j-1,k,l)*dyu(iLat-1)) /2./ dyt(iLat)
      adv_veu(i,j,k,l,iLat,iLon) =
     . ((adv_vet(i,j,k,l,iLat) * dus(iLat)
     . + adv_vet(i,j+1,k,l,iLat) * dun(iLat))*duw(iLon+1) +    
     .  (adv_vet(i+1,j,k,l,iLat) * dus(iLat)
     . + adv_vet(i+1,j+1,k,l,iLat) * dun(iLat))*due(iLon))
     .                     / dyu(iLat) / dxt(iLon+1)
      adv_fe (i,j,k,l,iLat,iLon) = adv_veu(i,j,k,l,iLat,iLon)
     .                  *(arg_1(i,j,k,l) + arg_1(i+1,j,k,l))


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

      l1 = arg_lo_ss(T_AXIS,ARG1) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
            ilat = 2  
            csu = cos(SNGL(yU(iLat)) * Degrees_to_radians) 
            j1 = arg_lo_ss(Y_AXIS,ARG1) + 1
            DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               iLon = 2
               i1 = arg_lo_ss(X_AXIS,ARG1) + 1
               DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
                  IF (arg_1(i1,  j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,  j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1+1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,  j1-1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1+1,j1-1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1-1,j1-1,k1,l1) .EQ. bad_flag(ARG1) 
     .                 )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) =
     .                  - (adv_fe(i1,j1,k1,l1,iLat,iLon) - 
     .                     adv_fe(i1-1,j1,k1,l1,iLat,iLon)) 
     .                       /2./dxu(iLon)/csu*sec_per_month
                  ENDIF
                  iLon = iLon + 1
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











