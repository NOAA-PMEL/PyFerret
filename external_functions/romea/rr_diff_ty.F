*           rr_diff_ty.F
*
*           Rick Romea
*           Jan. 24, 2000
*
*     Computes the meridional diffusive term: A T_yy
*     Units :     oC/s
*     MOM2 Grid:  T  
*
*******************************************************************

      SUBROUTINE RR_diff_ty_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'Ah(d^2T/dy^2) diffusive term (oC/s);MOM2 T Grid')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, NO, NO)

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
     .     'Potential temperature, on MOM2 T Grid.')
      CALL ef_set_arg_unit         (id, ARG2,  '(oC')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)
      END


      SUBROUTINE RR_diff_ty_compute(id, arg_1, arg_2, result)
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
      REAL*8         yT(1024)
      INTEGER        ilat
      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL dyt,dyu,xjt
      REAL cst,csu,diff_cnt,ahc_south,ahc_north
     .               
      INCLUDE 'rr_parameters.h'

C*********************************************************************
c  MOM2 Code:
c      DIFF_Ty(i,k,j) = ahc_north(j)*(t(i,k,j+1,n) - t(i,k,j,  n))
c     &               - ahc_south(j)*(t(i,k,j  ,n) - t(i,k,j-1,n))
c          ahc_north(j) = diff_cnt(j)*csu(j)*dyur(j)
c     &                            *cstr(j)*dytr(j)
c          ahc_south(j) = diff_cnt(j)*csu(j-1)*dyur(j-1)
c     &                            *cstr(j)*dytr(j)
c          diff_cnt(j)  = ah*xjt(j)
c          xjt(j) = 3.166*dyt(j)*dytr(50)-2.166
c     dyt     = latitudinal height of "t" grid box (in cm)
c     dytr   = reciprocal of "dyt"
c     dyu     = latitudinal height of "u,v" grid box (cm)
c     dyur    = reciprocal of "dyu"
c
c     ah     = constant lateral diffusion coeff for tracers(cm**2/sec) 
c               ah = 2.e7
c     csu     = cosine of "u,v" grid point latitude
c               csu(j)  = cos(phiu(j))
c     cst     = cosine of "t" grid point latitude
c               cst(j)  = cos(phit(j))
c     cstr    = reciprocal of "cst"
c     n = 1 => temperature
c     n = 2 => salinity
c*********************************************************************

      !  Statement functions

      dyt(iLat) = SNGL(yU(iLat)-yU(iLat-1))*Latitude_to_cm
      dyu(iLat) = SNGL(yT(iLat+1)-yT(iLat))*Latitude_to_cm
      xjt(iLat) = 3.166*dyt(iLat)/dyt_ref-2.166
      diff_cnt(iLat) = ah*xjt(iLat)
      cst(iLat) = cos(SNGL(yT(iLat))*Degrees_to_radians) 
      csu(iLat) = cos(SNGL(yU(iLat))*Degrees_to_radians) 
      ahc_south(iLat) = diff_cnt(iLat)*csu(iLat-1)
     .                         /cst(iLat)/dyu(iLat)/dyt(iLat)
      ahc_north(iLat) = diff_cnt(iLat)*csu(iLat)
     .                         /cst(iLat)/dyu(iLat)/dyt(iLat)
      ! Get axis data

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)
      CALL ef_get_coordinates    (id, ARG2, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG2),
     .                            arg_hi_ss(Y_AXIS,ARG2), yT)

      ! Check axis units:  bail out if not lat and lon.

      CALL ef_get_axis_info      (id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(1) .NE. 'deg'       .AND.
     .         ax_units(1) .NE. 'degrees_E'       .AND.
     .         ax_units(1) .NE. 'longitude' .AND.
     .         ax_units(1) .NE. 'Longitude' .AND. 
     .         ax_units(1) .NE. 'LONGITUDE' ) THEN
        WRITE (6,*)'Longitude axis units =', ax_units(1) 
        !CALL ef_bail_out(id,'Longitude axis must be in degrees')
      ENDIF

      IF     ( ax_units(2) .NE. 'deg'       .AND.
     .         ax_units(2) .NE. 'degrees_N'       .AND.
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
                     result(i,j,k,l) = (
     .                  ahc_north(iLat)*
     .                    (  arg_2(i2,j2+1,k2,l2)
     .                     - arg_2(i2,j2  ,k2,l2))
     .                - ahc_south(iLat)*
     .                    (  arg_2(i2,j2  ,k2,l2) 
     .                     - arg_2(i2,j2-1,k2,l2))
     .                             )*sec_per_month
                  ENDIF
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











