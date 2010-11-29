*           rr_diff_tx.F
*
*           Rick Romea
*           Jan. 24, 2000
*
*     Computes the zonal diffusive term: A T_xx
*     Units :     oC/s
*     MOM2 Grid:  T  
*
*******************************************************************

      SUBROUTINE RR_diff_tx_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'Ah d^2T/dx^2 diffusive term (oC/s);MOM2 T Grid')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .  IMPLIED_BY_ARGS,IMPLIED_BY_ARGS,IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok     (id, NO,NO,NO,NO)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, required to access the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)

      CALL ef_set_arg_name         (id, ARG2, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Potential temperature, on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'deg-C')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)
      END


      SUBROUTINE RR_diff_tx_compute(id, arg_1, arg_2, result)
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
      REAL dxt, dyt, dxu,xjt
      REAL cst,diff_fe
      INCLUDE 'rr_parameters.h'

C*********************************************************************
C
c  MOM2 Code:
c
c   DIFF_Tx(i,k,j) = (diff_fe(i,k,j) - diff_fe(i-1,k,j)) * cstdxtr(i,j)
c
c     diff_fe(i,k,j) = ah_cstdxur(i,j) * (t(i+1,k,j,n) - t(i,k,j,n))
c     cstdxtr(i,j)   = cstr(j) * dxtr(i)
c
c       ah_cstdxur(i,j) = diff_cet(j) * cstr(j) * dxur(i)
c
c          diff_cet(j)  = ah*xjt(j)
c
c             ah = constant lateral diffusion coeff for tracers
c                   ah = 2.e7(cm**2/sec)
c             xjt(j) = 3.166 * dyt(j) * dytr(50) - 2.166
c      
c     dxt     = longitudinal width of "t" grid box at the equator (cm)
c     dxtr    = reciprocal of "dxt"
c     dyt     = latitudinal height of "t" grid box (in cm)
c     dytr    = reciprocal of "dyt"
c     dxu     = longitudinal width of "u,v" grid box at the equator (cm) 
c     dxur    = reciprocal of "dxu"
c
c     cst     = cosine of "t" grid point latitude
c               cst(j)  = cos(phit(j))
c     cstr    = reciprocal of "cst"
c
c        n = 1 => temperature
c        n = 2 => salinity
c
c*********************************************************************

      dxt(i) = SNGL(xU(i)-xU(i-1))*Longitude_to_cm
      dyt(j) = SNGL(yU(j)-yU(j-1))*Latitude_to_cm
      dxu(i) = SNGL(xT(i+1)-xT(i))*Longitude_to_cm
      cst(i) = cos(SNGL(yT(i))*Degrees_to_radians) 
      xjt(i) = 3.166*dyt(i)/dyt_ref-2.166
      diff_fe(i,j,k,l,iLon,iLat) =ah*xjt(iLat) /cst(iLat)/dxu(iLon)
     .              *(arg_2(i+1,j,k,l)-arg_2(i,j,k,l))


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
      CALL ef_get_axis_info      (id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(1) .NE. 'deg'       .AND.
     .         ax_units(1) .NE. 'degrees_E' .AND.
     .         ax_units(1) .NE. 'longitude' .AND.
     .         ax_units(1) .NE. 'Longitude' .AND. 
     .         ax_units(1) .NE. 'LONGITUDE' ) THEN
        WRITE (6,*)'Longitude axis units =', ax_units(1) 
        !CALL ef_bail_out(id,'Longitude axis must be in degrees')
      ENDIF
      IF     ( ax_units(2) .NE. 'deg'       .AND.
     .         ax_units(2) .NE. 'degrees_N' .AND.
     .         ax_units(2) .NE. 'latitude'  .AND.
     .         ax_units(2) .NE. 'Latitude'  .AND. 
     .         ax_units(2) .NE. 'LATITUDE' ) THEN
        WRITE (6,*)'Latitude axis units =', ax_units(2) 
        !CALL ef_bail_out(id,'Latitude axis must be in degrees')
      ENDIF

      l2 = arg_lo_ss(T_AXIS,ARG2) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
            ilat = 2
            j2 = arg_lo_ss(Y_AXIS,ARG2) + 1 
            DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               iLon = 2
               i2 = arg_lo_ss(X_AXIS,ARG2) + 1
               DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
                  IF (arg_2(i2,  j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2,  j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2+1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_2(i2-1,j2-1,k2,l2) .EQ. bad_flag(ARG2) )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) =
     .                   (diff_fe(i2  ,j2,k2,l2,iLon,iLat)
     .                  - diff_fe(i2-1,j2,k2,l2,iLon-1,iLat))
     .                      /cst(iLat)/dxt(iLon)*sec_per_month
                  ENDIF
                  iLon = iLon + 1
                  i2 = i2 + arg_incr(X_AXIS,ARG2)
               ENDDO
               iLat = iLat + 1
               j2 = j2 + arg_incr(Y_AXIS,ARG2)
            ENDDO               
            k2 = k2 + arg_incr(Z_AXIS,ARG2)
         ENDDO
         l2 = l2 + arg_incr(T_AXIS,ARG2)
      ENDDO
      END











