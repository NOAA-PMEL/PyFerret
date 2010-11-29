*           rr_grad_pbc_v.F
*
*           Rick Romea
*           Jan. 24, 2000
*
*     Baroclinic pressure gradient term in v momentum equation
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
************************************************************************


	! NOTE: still need to fix up if first z grid point is not 1


      SUBROUTINE RR_grad_pbc_v_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'Barclininc -dp/dy term (cm/s^2) on the MOM2 U Grid')
      CALL ef_set_num_args         (id, 3)
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
      
      CALL ef_set_arg_name         (id, ARG2, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Potential Temperature, on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'oC')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,0,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,0,+1)

      
      CALL ef_set_arg_name         (id, ARG3, 'SALT')
      CALL ef_set_arg_desc         (id, ARG3,
     .  'Salinity on the MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG3, 'ppm-0.035')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG3, X_AXIS,0,+1)
      CALL ef_set_axis_extend      (id, ARG3, Y_AXIS,0,+1)

      END

c*******************************************************************

      SUBROUTINE RR_grad_pbc_v_compute(id, arg_1, arg_2, arg_3, result)
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
      REAL arg_3  (mem3lox:mem2hix, mem3loy:mem2hiy, 
     .             mem3loz:mem2hiz, mem3lot:mem2hit)
      REAL result (memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit)

      INTEGER  res_lo_ss (4)
      INTEGER  res_hi_ss (4)
      INTEGER  res_incr  (4)

      INTEGER  arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER  arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER  arg_incr  (4,EF_MAX_ARGS)

      INTEGER i,  j,  k,  l, m
      INTEGER i2, j2, k2, l2, m1
      INTEGER i3, j3, k3, l3

      REAL*8         xU(1024)
      REAL*8         yU(1024)
      REAL*8         zU(1024)

      INTEGER        ilat
      INTEGER        ilon
      INTEGER        iZ

      CHARACTER*16  ax_name(4)
      CHARACTER*16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL       dyt  
      REAL*8     Rho  ! functions for density
      REAL       ss,tt

      INCLUDE 'rr_parameters.h'

      ! Statement Functions

      ss(i,j,k,l) =  arg_3(i,j,k,l)*1000. + 35. ! True salinity (ppt)
      tt(i,j,k,l) =  arg_2(i,j,k,l)

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)

	! Check that upper z grid point is the surface z grid point
	
      IF (arg_lo_ss(Z_AXIS,ARG1) .NE. 1) THEN
        CALL ef_bail_out(id,'U:Upper grid point must be surface')
	ENDIF
      IF (arg_lo_ss(Z_AXIS,ARG2) .NE. 1) THEN
        CALL ef_bail_out(id,'TEMP:Upper grid point must be surface')
      ENDIF
      IF (arg_lo_ss(Z_AXIS,ARG3) .NE. 1) THEN
        CALL ef_bail_out(id,'SALT:Upper grid point must be surface')
	ENDIF

      ! Get U grid coordinates

      CALL ef_get_coordinates    (id, ARG1, X_AXIS, 
     .                            arg_lo_ss(X_AXIS,ARG1),
     .                            arg_hi_ss(X_AXIS,ARG1), xU)    
      CALL ef_get_coordinates    (id, ARG1, Y_AXIS, 
     .                            arg_lo_ss(Y_AXIS,ARG1),
     .                            arg_hi_ss(Y_AXIS,ARG1), yU)
      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG1),
     .                            arg_hi_ss(Z_AXIS,ARG1), zU)

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
      j2 = arg_lo_ss(Y_AXIS,ARG2) 
      j3 = arg_lo_ss(Y_AXIS,ARG3) 
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         dyt = SNGL (yU(iLat+1)-yU(iLat)) * Latitude_to_cm
         iLon = 1
         i2 = arg_lo_ss(X_AXIS,ARG2) 
         i3 = arg_lo_ss(X_AXIS,ARG3) 
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            l2 = arg_lo_ss(T_AXIS,ARG2)
            l3 = arg_lo_ss(T_AXIS,ARG3)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

	       iZ = 1
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  IF (
     .              arg_2(i2,  j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2,  j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2,  j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2+1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2+1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2+1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2-1,j2,  k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2-1,j2+1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_2(i2-1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .              arg_3(i3,  j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3,  j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3,  j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3+1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3+1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3+1,j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3-1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3-1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .              arg_3(i3-1,j3-1,k3,l3) .EQ. bad_flag(ARG3) 
     .                      ) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE

                    IF (iZ .EQ. 1) THEN  ! Top z-grid point.
                      
                      result(i,j,k,l) = SNGL(zU(iZ)) * SNGL( 
     .  Rho(DBLE(tt(i2+1,j2+1,k2,l2)),DBLE(ss(i3+1,j3+1,k3,l3)),zU(iZ)) 
     . +Rho(DBLE(tt(i2  ,j2+1,k2,l2)),DBLE(ss(i3  ,j3+1,k3,l3)),zU(iZ)) 
     . -Rho(DBLE(tt(i2+1,j2  ,k2,l2)),DBLE(ss(i3+1,j3  ,k3,l3)),zU(iZ)) 
     . -Rho(DBLE(tt(i2  ,j2  ,k2,l2)),DBLE(ss(i3  ,j3  ,k3,l3)),zU(iZ))
     .                         )
                     ELSE
                       result(i,j,k,l) = 0.
					  
		       m1=arg_lo_ss(Z_AXIS,ARG2)+1
                       DO m=2,iZ
                          result(i,j,k,l) = result(i,j,k,l) + 
     .                             SNGL (zU(m+1) - zU(m)) * SNGL ( 
     .            Rho(DBLE(tt(i2+1,j2+1,m1  ,l2)),
     .                DBLE(ss(i3+1,j3+1,m1  ,l3)),zU(m  ))
     .           +Rho(DBLE(tt(i2+1,j2+1,m1-1,l2)),
     .                DBLE(ss(i3+1,j3+1,m1-1,l3)),zU(m-1))
     .           +Rho(DBLE(tt(i2  ,j2+1,m1  ,l2)),
     .                DBLE(ss(i3  ,j3+1,m1  ,l3)),zU(m  ))
     .           +Rho(DBLE(tt(i2  ,j2+1,m1-1,l2)),
     .                DBLE(ss(i3  ,j3+1,m1-1,l3)),zU(m-1))
     .           -Rho(DBLE(tt(i2+1,j2  ,m1  ,l2)),
     .                DBLE(ss(i3+1,j3  ,m1  ,l3)),zU(m  ))
     .           -Rho(DBLE(tt(i2+1,j2  ,m1-1,l2)),
     .                DBLE(ss(i3+1,j3  ,m1-1,l3)),zU(m-1))
     .           -Rho(DBLE(tt(i2  ,j2  ,m1  ,l2)),
     .                DBLE(ss(i3  ,j3  ,m1  ,l3)),zU(m  ))
     .           -Rho(DBLE(tt(i2  ,j2  ,m1-1,l2)),
     .                DBLE(ss(i3  ,j3  ,m1-1,l3)),zU(m-1))
     .               )
	               m1 = m1 + arg_incr(Z_AXIS,ARG2)
                       ENDDO
                       result(i,j,k,l) = result(i,j,k,l)/2.
                     ENDIF
                     result(i,j,k,l) = result(i,j,k,l)*RR_temp1/dyt
     .                                   * sec_per_month

                  ENDIF

                 iZ = iZ + 1
                 k2 = k2 + arg_incr(Z_AXIS,ARG2)
                 k3 = k3 + arg_incr(Z_AXIS,ARG3)
                 ENDDO

               l2 = l2 + arg_incr(T_AXIS,ARG2)
               l3 = l3 + arg_incr(T_AXIS,ARG3)
            ENDDO

            iLon = iLon + 1
            i2 = i2 + arg_incr(X_AXIS,ARG2)
            i3 = i3 + arg_incr(X_AXIS,ARG3)
         ENDDO

         iLat = iLat + 1
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
      ENDDO
      END


      ! Stolen KNUDSEN code to compute density

      ! D = depth (m)
      ! T = temperature (oC)
      ! S = salinity (ppt)

      REAL*8 FUNCTION Rho(T,S,D)
      IMPLICIT NONE
      REAL*8 T,S,D
      REAL*8 T2,T3,S2,S3,F1,F2,F3,FS,SIGMA,A,B1,B2,B,CO,ALPHA
      T2=T*T
      T3=T2*T
      S2=S*S
      S3=S2*S
      F1=-(T-3.98d0)**2*(T+283.d0)/(503.57d0*(T+67.26d0))
      F2=T3*1.0843d-6-T2*9.8185d-5+T*4.786d-3
      F3=T3*1.667d-8-T2*8.164d-7+T*1.803d-5
      FS=S3*6.76786136d-6-S2*4.8249614d-4+S*8.14876577d-1
      SIGMA=F1+(FS+3.895414d-2)*(1.d0-F2+F3*(FS-.22584586d0))
      A=D*1.d-4*(105.5d0+T*9.5d0-T2*.158d0-D*T*1.5d-4)-
     .(227.d0+T*28.33d0-T2*.551d0+T3*.004d0)
      B1=(FS-28.1324d0)/10.d0
      B2=B1*B1
      B=-B1*(147.3d0-T*2.72d0+T2*.04d0-D*1.d-
     .4*(32.4d0-.87d0*T+.02d0*T2))
      B=B+B2*(4.5d0-.1d0*T-D*1.d-4*(1.8d0-.06d0*T))
      CO=4886.d0/(1.d0+1.83d-5*D)
      ALPHA=D*1.d-6*(CO+A+B)
      Rho=(SIGMA+ALPHA)/(1.d0-1.d-3*ALPHA)
      END

