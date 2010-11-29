*     rr_diff_uz.F
*
*           Rick Romea
*           Jan. 24, 2000 
*
*     Zonal momentum diffusive term :  [Az u_z]_z
*     Units :     cm/s/s
*     MOM2 Grid :  U   
*
*******************************************************************

      SUBROUTINE RR_diff_uz_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'d(Az du/dz)/dz momentum diffusive term (cm/s^2);MOM2 U Grid')
      CALL ef_set_num_args         (id, 5)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, NO, NO)

      ! Define first argument (U)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Z_AXIS,0,+1)

      ! Define second argument (V)

      CALL ef_set_arg_name         (id, ARG2, 'V')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Meridional velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2,YES,YES,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Z_AXIS,0,+1)

      ! Define third argument (TEMP)

      CALL ef_set_arg_name         (id, ARG3, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG3,
     . 'Potential temperature, on MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG3, 'deg-C')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG3, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Z_AXIS,0,+1)

      ! Define forth argument (SALT)

      CALL ef_set_arg_name         (id, ARG4, 'SALT')
      CALL ef_set_arg_desc         (id, ARG4,
     . 'Salinity, on MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG4, '(ppt-35)/1000')
      CALL ef_set_arg_type         (id, ARG4, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG4, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG4, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG4, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG4, Z_AXIS,0,+1)

      ! Define fifth argument (TAUX)

      CALL ef_set_arg_name         (id, ARG5, 'TAUX')
      CALL ef_set_arg_desc         (id, ARG5,
     . 'Zonal wind stress.')
      CALL ef_set_arg_unit         (id, ARG5, 'dynes/cm^2')
      CALL ef_set_arg_type         (id, ARG5, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG5, NO, NO, NO,YES)
      CALL ef_set_axis_extend      (id, ARG5, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG5, Y_AXIS,-1,+1)

      ! Define a work array

      CALL ef_set_num_work_arrays  (id, 1)

      END 

C*************************************************************************

      SUBROUTINE RR_diff_uz_work_size(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id
      INTEGER res_lo_ss (4)
      INTEGER res_hi_ss (4)
      INTEGER res_incr  (4)
      CALL ef_get_res_subscripts  (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_set_work_array_dims (id, 1,  
     .    res_lo_ss(1),  res_lo_ss(2), res_lo_ss(3),  res_lo_ss(4),
     .    res_hi_ss(1),  res_hi_ss(2), res_hi_ss(3),  res_hi_ss(4) )
      END

C*************************************************************************

      SUBROUTINE RR_diff_uz_compute(id, 
     .             arg_1,arg_2,arg_3,arg_4,arg_5,result,wrk_1)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      
      INTEGER        id

      REAL           bad_flag(EF_MAX_ARGS)
      REAL           bad_flag_result

      REAL           arg_1  (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .                       mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL           arg_2  (mem2lox:mem2hix, mem2loy:mem2hiy, 
     .                       mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL           arg_3  (mem3lox:mem3hix, mem3loy:mem3hiy, 
     .                       mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL           arg_4  (mem4lox:mem4hix, mem4loy:mem4hiy, 
     .                       mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL           arg_5  (mem5lox:mem5hix, mem5loy:mem5hiy, 
     .                       mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL           result (memreslox:memreshix,memresloy:memreshiy,
     .                       memresloz:memreshiz,memreslot:memreshit)
      REAL           wrk_1  (memreslox:memreshix,memresloy:memreshiy,
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
      INTEGER        i3, j3, k3, l3
      INTEGER        i4, j4, k4, l4
      INTEGER        i5, j5, k5, l5

      REAL*8         zT(1024)
      REAL*8         zU(1024)
  
      INTEGER        iZ

      REAL get_dzw,dzw,dzt,diff_fb,rhom1z,Ri,Potential_density

      INCLUDE 'rr_parameters.h'


c*********************************************************************
c     THIS IS THE MOM2 CODE:
c
c     dzt     = thickness of "t" grid cells (cm)
c     dztr(k) = reciprocal of "dzt"
c     dzw(k)  = vertical resolution of "w" grid cells (in cm)
c     dzwr(k) = reciprocal of "dzw"
c     visc_cbu       = viscosity coefficient at bottom of "u" cells (cm**2/s)
c     wndmix         = min value for mixing at surface to 
c                     simulate high freq wind mixing. (cm**2/sec)  = 10.
c     smf     = surface momentum flux (dynes/cm**2)
c                smf(1) = taux     smf(2) = tauy
c     grav    = gravity (cm/sec**2)
c     epsln   = small value: 1.e-25
c
c      DIFF_Uz(i,k,j) = (diff_fb(i,k-1,j)-diff_fb(i,k,j))*dztr(k)
c      diff_fb(i,k,j) = visc_cbu(i,k,j)*dzwr(k)*(u(i,k,j,n)-u(i,k+1,j,n))
c      diff_fb(i,0,j) = smf(i,j,n) ! taux (dynes/cm^3)
c      t1                = 1./(1. + 5.*riu(i,k,j))
c      visc_cbu(i,k,j)   = fricmx*t1**2 + visc_cbu_back
c      if(riu(i,k,j).lt.0.)visc_cbu(i,k,j)=visc_cbu_limit ! unstable
c      if(visc_cbu(i,1,j).lt.wndmix)visc_cbu(i,1,j)=wndmix ! surface
c
c      riu(i,k,j) = -grav/4.*dzw(k)*(rhom1z(i,k,j+1) + rhom1z(i+1,k,j+1) +
c     &                            rhom1z(i,k,j)   + rhom1z(i+1,k,j)) /
c     &  ((u(i,k,j,1)-u(i,k+1,j,1))**2+(u(i,k,j,2)-u(i,k+1,j,2))**2 +  epsln) 
c      rhom1z(i,k,j) = ro(i,k,j) - ro(i,k+1,j)
c
c      do ks=1,2
c        call statec (t(1,1,1,1,tlev), t(1,1,1,2,tlev), ro(1,1,jsmw)
c     &,              max(js,jsmw), je, istrt-1, iend+1, ks)
c*********************************************************************

      !  Statement functions

      dzw(iZ) =  get_dzw(iZ,zT) * Meters_to_cm
      dzt(iZ) = (dzw(iZ-1) + dzw(iZ)) / 2.
      diff_fb(i,j,k,l,iZ) = wrk_1(i,j,k,l)/dzw(iZ)  ! wrk_1=visc_cbu
     .                 *(arg_1(i,j,k,l)-arg_1(i,j,k+1,l))
      rhom1z(i3,j3,k3,l3,i4,j4,k4,l4) =
     .        Potential_density(arg_3(i3,j3,k3  ,l3),
     .                          arg_4(i4,j4,k4  ,l4))
     .      - Potential_density(arg_3(i3,j3,k3+1,l3),
     .                          arg_4(i4,j4,k4+1,l4)) 

      ! Get axis data

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG1),
     .                            arg_hi_ss(Z_AXIS,ARG1), zU)
      CALL ef_get_coordinates    (id, ARG3, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG3),
     .                            arg_hi_ss(Z_AXIS,ARG3), zT)

       ! Compute the viscosity coefficients

      l1 = arg_lo_ss(T_AXIS,ARG1) 
      l2 = arg_lo_ss(T_AXIS,ARG2) 
      l3 = arg_lo_ss(T_AXIS,ARG3) 
      l4 = arg_lo_ss(T_AXIS,ARG4) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1) + 1
         j2 = arg_lo_ss(Y_AXIS,ARG2) + 1
         j3 = arg_lo_ss(Y_AXIS,ARG3) + 1
         j4 = arg_lo_ss(Y_AXIS,ARG4) + 1
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1) + 1
            i2 = arg_lo_ss(X_AXIS,ARG2) + 1
            i3 = arg_lo_ss(X_AXIS,ARG3) + 1
            i4 = arg_lo_ss(X_AXIS,ARG4) + 1
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               iZ = 1
               k1 = arg_lo_ss(Z_AXIS,ARG1)
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               k4 = arg_lo_ss(Z_AXIS,ARG4)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                 
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
     .                arg_2(i2-1,j2-1,k2,l2) .EQ. bad_flag(ARG2) .OR.
     .                arg_3(i3,  j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3,  k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3,  j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3+1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3,  j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3+1,j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_3(i3-1,j3-1,k3,l3) .EQ. bad_flag(ARG3) .OR.
     .                arg_4(i4,  j4,  k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4+1,j4,  k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4-1,j4,  k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4,  j4+1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4+1,j4+1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4-1,j4+1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4,  j4-1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4+1,j4-1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                arg_4(i4-1,j4-1,k4,l4) .EQ. bad_flag(ARG4)
     .                 )THEN
                     wrk_1(i,j,k,l) = bad_flag_result
                  ELSE
                     Ri = - gravity/4. * dzw(iZ) * (
     .                  rhom1z(i3  ,j3+1,k3,l3,i4  ,j4+1,k4,l4) + 
     .                  rhom1z(i3+1,j3+1,k3,l3,i4+1,j4+1,k4,l4) +
     .                  rhom1z(i3  ,j3  ,k3,l3,i4  ,j4  ,k4,l4) + 
     .                  rhom1z(i3+1,j3  ,k3,l3,i4+1,j4  ,k4,l4) ) /
     .               ( (arg_1(i1,j1,k1,l1)-arg_1(i1,j1,k1+1,l1))**2 +
     .                 (arg_2(i2,j2,k2,l2)-arg_2(i2,j2,k2+1,l2))**2 
     .                        + epsln )

                     ! Viscosity coefficient

                     IF(Ri.lt.0.)THEN
                        wrk_1(i,j,k,l) = visc_cbu_limit  ! unstable
                     ELSE
                        wrk_1(i,j,k,l) = fricmx/(1.+5.*Ri)**2
     .                                      + visc_cbu_back
                     ENDIF
                     IF(k1.EQ.1.AND.wrk_1(i,j,k,l).lt.wndmix)THEN
                        wrk_1(i,j,k,l)=wndmix ! surface
                     ENDIF
                  ENDIF

                  iZ = iZ + 1 
                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
                  k3 = k3 + arg_incr(Z_AXIS,ARG3)
                  k4 = k4 + arg_incr(Z_AXIS,ARG4)
               ENDDO

               i1 = i1 + arg_incr(X_AXIS,ARG1)
               i2 = i2 + arg_incr(X_AXIS,ARG2)
               i3 = i3 + arg_incr(X_AXIS,ARG3)
               i4 = i4 + arg_incr(X_AXIS,ARG4)
            ENDDO

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
         ENDDO               

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
      ENDDO

      ! Now compute the term
      
      k5 = arg_lo_ss(Z_AXIS,ARG5)
      l1 = arg_lo_ss(T_AXIS,ARG1) 
      l5 = arg_lo_ss(T_AXIS,ARG5)
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1) + 1
         j5 = arg_lo_ss(Y_AXIS,ARG5) + 1
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1) + 1
            i5 = arg_lo_ss(X_AXIS,ARG5) + 1
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               iZ = 1
               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  IF(wrk_1(i,j,k,l).EQ. bad_flag_result)THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSEIF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,j1,k1+1,l1) .EQ. bad_flag(ARG1) )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE

                    IF (k.EQ.1) THEN  ! Top term is wind stress term
                       IF (arg_5(i5,j5,k5,l5) .EQ. bad_flag(ARG5))THEN
                          result(i,j,k,l) = bad_flag_result
                       ELSE
                          result(i,j,k,l) = (   arg_5(i5,j5,k5,l5)
     .                                  -  diff_fb(i1,j1,k1,l1,iZ)  )
     .                                     /dzt(iZ) * sec_per_month 
                       ENDIF
                    ELSE
                       result(i,j,k,l) = ( diff_fb(i1,j1,k1-1,l1,iZ-1)
     .                                  -  diff_fb(i1,j1,k1,l1,iZ) )
     .                                     /dzt(iZ) * sec_per_month 
                    ENDIF

                  ENDIF

                  iZ = iZ + 1 
                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
               ENDDO

               i1 = i1 + arg_incr(X_AXIS,ARG1)
               i5 = i5 + arg_incr(X_AXIS,ARG5)
            ENDDO

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j5 = j5 + arg_incr(Y_AXIS,ARG5)
         ENDDO               

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l5 = l5 + arg_incr(T_AXIS,ARG5)
      ENDDO

      END


      ! Density at one Bar 

      REAL FUNCTION Potential_density(T,S)
      IMPLICIT NONE
      REAL T ! Potential Temperature  
      REAL S ! ( SALINITY(ppt) - 35) / 1000 
      REAL Pressure / 1.0 / ! Bar
      REAL Sppt ! Salinity (ppt)     
      Sppt = 1.e3*S+35.  ! Convert S: -->  ppt
      CALL RR_unesco(T,Sppt,Pressure,Potential_density)
      Potential_density = Potential_density*1.e-3  ! Convert:kg/m^3-->g/cm^3
      END


	REAL FUNCTION get_dzw(iZ,zT) 
      IMPLICIT NONE
	INTEGER iZ
      REAL*8 zT(iZ+1) 
      IF(iZ.EQ.0)THEN
	    get_dzw=SNGL(zT(1))
      ELSE
        get_dzw=SNGL(zT(iZ+1)-zT(iZ))
      ENDIF
	END









