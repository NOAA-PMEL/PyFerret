*     rr_av.F   ----    Rick Romea     Jun. 6, 2000
*
*     Vertical diffusion coefficient: Az (Units = cm^2/s)
*
*******************************************************************

      SUBROUTINE RR_av_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id
      CALL ef_set_desc(id,'Vertical diffusion coefficient:Az (cm^2/s)')
      CALL ef_set_num_args(id,4)
      CALL ef_set_axis_inheritance(id,IMPLIED_BY_ARGS, 
     . IMPLIED_BY_ARGS,IMPLIED_BY_ARGS,IMPLIED_BY_ARGS)

      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG1, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES, YES, YES, YES)
      CALL ef_set_axis_extend      (id, ARG1, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG1, Z_AXIS,0,+1)

      CALL ef_set_arg_name         (id, ARG2, 'V')
      CALL ef_set_arg_desc         (id, ARG2,
     . 'Meridional velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, ARG2, 'cm/sec')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2,YES,YES,YES,YES)
      CALL ef_set_axis_extend      (id, ARG2, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG2, Z_AXIS,0,+1)

      CALL ef_set_arg_name         (id, ARG3, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG3,
     . 'Potential temperature, on MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG3, 'deg-C')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG3, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG3, Z_AXIS,0,+1)

      CALL ef_set_arg_name         (id, ARG4, 'SALT')
      CALL ef_set_arg_desc         (id, ARG4,
     . 'Salinity, on MOM2 T Grid. ')
      CALL ef_set_arg_unit         (id, ARG4, '(ppt-35)/1000')
      CALL ef_set_arg_type         (id, ARG4, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG4, NO, NO,YES,YES)
      CALL ef_set_axis_extend      (id, ARG4, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG4, Y_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG4, Z_AXIS,0,+1)

      END 


      SUBROUTINE RR_av_compute(id,arg_1,arg_2,arg_3,arg_4,result)
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
      INTEGER        i3, j3, k3, l3
      INTEGER        i4, j4, k4, l4

      REAL*8         zT(1024)
      INTEGER        iZ
      REAL get_dzw,dzw,dzt,rhom1z,Ri,Potential_density

      INCLUDE 'rr_parameters.h'

      ! visc_cbu_back = 1.
      ! wndmix = 10.
      ! visc_cbu_limit = fricmx
      ! fricmx = 50.
      ! gravity = 980.6

c*********************************************************************
c     THIS IS THE MOM2 CODE:
c
c     dzt     = thickness of "t" grid cells (cm)
c     dztr(k) = reciprocal of "dzt"
c     dzw(k)  = vertical resolution of "w" grid cells (in cm)
c     dzwr(k) = reciprocal of "dzw"
c     visc_cbu       = viscosity coefficient at bottom of "u" cells (cm**2/s)
c     visc_cbu_back  = background viscosity (cm**2/s)  =  1.
c     fricmx         = maximum viscosity    (cm**2/s)  = 50.
c     visc_cbu_limit = largest viscosity (cm**2/sec) = fricmx
c     wndmix         = min value for mixing at surface to 
c                     simulate high freq wind mixing. (cm**2/sec)  = 10.
c     smf     = surface momentum flux (dynes/cm**2)
c                smf(1) = taux     smf(2) = tauy
c     grav    = gravity (cm/sec**2)
c     epsln   = small value: 1.e-25
c
c      t1                = 1./(1. + 5.*riu(i,k,j))
c      visc_cbu(i,k,j)   = fricmx*t1**2 + visc_cbu_back
c      if(riu(i,k,j).lt.0.)visc_cbu(i,k,j)=visc_cbu_limit ! unstable
c      if(visc_cbu(i,1,j).lt.wndmix)visc_cbu(i,1,j)=wndmix ! surface
c
c      riu(i,k,j) = -grav/4.*dzw(k)*(rhom1z(i,k,j+1) + rhom1z(i+1,k,j+1) +
c     &                            rhom1z(i,k,j)   + rhom1z(i+1,k,j)) /
c     &  ((u(i,k,j,1)-u(i,k+1,j,1))**2+(u(i,k,j,2)-u(i,k+1,j,2))**2 +  epsln) 
c      rhom1z(i,k,j) = ro(i,k,j) - ro(i,k+1,j)
c      do ks=1,2
c        call statec (t(1,1,1,1,tlev), t(1,1,1,2,tlev), ro(1,1,jsmw)
c     &,              max(js,jsmw), je, istrt-1, iend+1, ks)
c*********************************************************************

      !  Statement functions

      dzw(iZ) =  get_dzw(iZ,zT) * Meters_to_cm
      dzt(iZ) = (dzw(iZ-1) + dzw(iZ)) / 2.
      rhom1z(i3,j3,k3,l3,i4,j4,k4,l4) =
     .        Potential_density(arg_3(i3,j3,k3  ,l3),
     .                          arg_4(i4,j4,k4  ,l4))
     .      - Potential_density(arg_3(i3,j3,k3+1,l3),
     .                          arg_4(i4,j4,k4+1,l4)) 

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG3, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG3),
     .                            arg_hi_ss(Z_AXIS,ARG3), zT)

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
                  IF (arg_1(i1,  j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
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
     .                  )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     Ri = - gravity/4. * dzw(iZ) * (
     .                  rhom1z(i3  ,j3+1,k3,l3,i4  ,j4+1,k4,l4) + 
     .                  rhom1z(i3+1,j3+1,k3,l3,i4+1,j4+1,k4,l4) +
     .                  rhom1z(i3  ,j3  ,k3,l3,i4  ,j4  ,k4,l4) + 
     .                  rhom1z(i3+1,j3  ,k3,l3,i4+1,j4  ,k4,l4) ) /
     .               ( (arg_1(i1,j1,k1,l1)-arg_1(i1,j1,k1+1,l1))**2 +
     .                 (arg_2(i2,j2,k2,l2)-arg_2(i2,j2,k2+1,l2))**2 
     .                        + epsln )
                     IF(Ri.lt.0.25)THEN
                        result(i,j,k,l) = visc_cbu_limit  ! unstable
                     ELSE
                        result(i,j,k,l) = fricmx/(1.+5.*Ri)**2
     .                                      + visc_cbu_back
                     ENDIF
                     IF(k1.EQ.1.AND.result(i,j,k,l).lt.wndmix)
     .                   result(i,j,k,l)=wndmix ! surface
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
      END


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












