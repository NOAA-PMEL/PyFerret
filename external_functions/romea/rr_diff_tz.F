
*     rr_diff_tz.F
*
*           Rick Romea
*           Jan. 24, 2000  
*
*     Zonal momentum diffusive term :  [Az T_z]_z
*     Units :     oC/s
*     MOM2 Grid :  T   
*
*******************************************************************

      SUBROUTINE RR_diff_tz_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,
     .'d(Az dT/dz)/dz momentum diffusive term (oC/s);MOM2 T Grid')
      CALL ef_set_num_args         (id, 5)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, NO, NO, NO, NO)

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

      CALL ef_set_arg_name         (id, ARG5, 'HFLX')
      CALL ef_set_arg_desc         (id, ARG5,
     . 'Surface heat flux.')
      CALL ef_set_arg_unit         (id, ARG5, 'cal/cm^2/sec')
      CALL ef_set_arg_type         (id, ARG5, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG5, NO, NO, NO,YES)
      CALL ef_set_axis_extend      (id, ARG5, X_AXIS,-1,+1)
      CALL ef_set_axis_extend      (id, ARG5, Y_AXIS,-1,+1)

      END 


      SUBROUTINE RR_diff_tz_compute(id, 
     .             arg_1,arg_2,arg_3,arg_4,arg_5,result)
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

      REAL get_dzw,diff_fb,diff_cbt
      REAL rhom1z,riu,Ri,Potential_density

      INCLUDE 'rr_parameters.h'

c  MOM2:  diff_fb(i,k,j)=diff_cbt(i,k,j)/dzw(k)*(t(i,k,j)-t(i,k+1,j))
c         diff_cbt = diffusion coefficient at bottom of "t" cells (cm**2/s)
c         dzw(k) = zT(k+1)-zT(k)

      diff_fb(i,j,k,l,iZ) = diff_cbt/get_dzw(iZ,zT) 
     .                 *(arg_3(i,j,k,l)-arg_3(i,j,k+1,l)) ! arg_3 = T

c  MOM2:  rhom1z(i,k,j) = ro(i,k,j) - ro(i,k+1,j)

      rhom1z(i3,j3,k3,l3,i4,j4,k4,l4) =
     .        Potential_density(arg_3(i3,j3,k3  ,l3),   ! arg_3 = T
     .                          arg_4(i4,j4,k4  ,l4))   ! arg_4 = S
     .      - Potential_density(arg_3(i3,j3,k3+1,l3),
     .                          arg_4(i4,j4,k4+1,l4)) 

c  MOM2:    grav    = gravity (cm/sec**2)
c  MOM2:    epsln   = small value: 1.e-25
     
      ! rr_parameters.h :
      !    REAL gravity = 980.6  cm/sec**2 
      !    REAL epsln = 1.e-25
 
c  MOM2:
c     riu(i,k,j) = -grav/4.*dzw(k)*(rhom1z(i,k,j+1) + rhom1z(i+1,k,j+1) +
c    &                            rhom1z(i,k,j)   + rhom1z(i+1,k,j)) /
c    &  ((u(i,k,j,1)-u(i,k+1,j,1))**2+(u(i,k,j,2)-u(i,k+1,j,2))**2 +  epsln) 

      riu (i1,j1,k1,l1,
     .     i2,j2,k2,l2,
     .     i3,j3,k3,l3,
     .     i4,j4,k4,l4, iZ) =  - gravity/4. * get_dzw(iZ,zT) * (
     .                  rhom1z(i3  ,j3+1,k3,l3,i4  ,j4+1,k4,l4) + 
     .                  rhom1z(i3+1,j3+1,k3,l3,i4+1,j4+1,k4,l4) +
     .                  rhom1z(i3  ,j3  ,k3,l3,i4  ,j4  ,k4,l4) + 
     .                  rhom1z(i3+1,j3  ,k3,l3,i4+1,j4  ,k4,l4) ) /
     .               ( (arg_1(i1,j1,k1,l1)-arg_1(i1,j1,k1+1,l1))**2 + !arg_1=U
     .                 (arg_2(i2,j2,k2,l2)-arg_2(i2,j2,k2+1,l2))**2   !arg_2=V
     .                        + epsln )

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG1),
     .                            arg_hi_ss(Z_AXIS,ARG1), zU)
      CALL ef_get_coordinates    (id, ARG3, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG3),
     .                            arg_hi_ss(Z_AXIS,ARG3), zT)

      l1 = arg_lo_ss(T_AXIS,ARG1) 
      l2 = arg_lo_ss(T_AXIS,ARG2) 
      l3 = arg_lo_ss(T_AXIS,ARG3) 
      l4 = arg_lo_ss(T_AXIS,ARG4) 
      l5 = arg_lo_ss(T_AXIS,ARG5)
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         j1 = arg_lo_ss(Y_AXIS,ARG1) + 1
         j2 = arg_lo_ss(Y_AXIS,ARG2) + 1
         j3 = arg_lo_ss(Y_AXIS,ARG3) + 1
         j4 = arg_lo_ss(Y_AXIS,ARG4) + 1
         j5 = arg_lo_ss(Y_AXIS,ARG5) + 1
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            i1 = arg_lo_ss(X_AXIS,ARG1) + 1
            i2 = arg_lo_ss(X_AXIS,ARG2) + 1
            i3 = arg_lo_ss(X_AXIS,ARG3) + 1
            i4 = arg_lo_ss(X_AXIS,ARG4) + 1
            i5 = arg_lo_ss(X_AXIS,ARG5) + 1
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               iZ = 1
               k1 = arg_lo_ss(Z_AXIS,ARG1)
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               k4 = arg_lo_ss(Z_AXIS,ARG4)
               k5 = arg_lo_ss(Z_AXIS,ARG5)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  IF (arg_1(i1,  j1,  k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,  j1,k1+1,l1) .EQ. bad_flag(ARG1) .OR.
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
     .                arg_4(i4-1,j4-1,k4,l4) .EQ. bad_flag(ARG4) .OR.
     .                (iZ .EQ. 1 .AND. 
     .                    arg_5(i5,j5,k5,l5).EQ. bad_flag(ARG5))
     .                  )THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE



c  MOM2:   rit(i,k,j)=(riu(i,k,j)+riu(i-1,k,j)+riu(i,k,j-1)+riu(i-1,k,j-1))/4

                     Ri = (Riu(i1,  j1,  k1,l1, i2,  j2,  k2,l2,
     .                         i3,  j3,  k3,l3, i4,  j4,  k4,l4,iZ) 
     .                 +   Riu(i1-1,j1,  k1,l1, i2-1,j2,  k2,l2,
     .                         i3-1,j3,  k3,l3, i4-1,j4,  k4,l4,iZ) 
     .                 +   Riu(i1,  j1-1,k1,l1, i2,  j2-1,k2,l2,
     .                         i3,  j3-1,k3,l3, i4,  j4-1,k4,l4,iZ) 
     .                 +   Riu(i1-1,j1-1,k1,l1, i2-1,j2-1,k2,l2,
     .                         i3-1,j3-1,k3,l3, i4-1,j4-1,k4,l4,iZ) 
     .                      )/4.



             ! Compute diffusion coefficient

             ! REAL fricmx=maximum diffusion coefficient=50 cm**2/s 
             ! REAL diff_cbt_limit=largest diffusion coefficient=50 cm**2/sec 
             ! REAL diff_cbt_back=background diffusion coefficient=0.1 cm**2/s
             ! REAL wndmix = min value for diffusion coefficient at surface to 
             !               simulate high freq wind mixing = 10 cm**2/sec 
             ! REAL Meters_to_cm = Convert: m --> cm = 100.
             ! REAL sec_per_month = Time conversion :  Convert: month --> sec
             !                      NOTE: sec_per_month currently set = 1, 
             !                      so the function returns 
             !                      oC/s, not oC/month

c MOM2:   check stability: if(rit(i,k,j).lt.0.)diff_cbt(i,k,j,1)=diff_cbt_limit
c                      diff_cbt(i,k,j,1) = fricmx*t2**3 + diff_cbt_back
c                               t2 = 1/(1 + 5*rit(i,k,j))
c         surface effect at top grid point:
c                    if(diff_cbt(i,1,j,1).lt.wndmix)diff_cbt(i,1,j,1)=wndmix

                     IF(Ri.lt.0.)THEN
                        ! PRINT*,'WARNING::Ri<0.-->unstable'
                        diff_cbt = diff_cbt_limit  ! unstable
                     ELSE
                        diff_cbt = fricmx/(1.+5.*Ri)**3
     .                                      + diff_cbt_back
                        IF(iZ.EQ.1.AND.diff_cbt.lt.wndmix)THEN
                           !PRINT*,'surface flux pegged at wndmix'
                           diff_cbt=wndmix   ! surface
                        ENDIF
                     ENDIF


c  MOM2:
c      DIFF_Tz(i,k,j) = (diff_fb(i,k-1,j) - diff_fb(i,k,j)) / dzt(k)
c         diff_fb(i,0,j)  = stf(i,j,n)
c     dzt    = thickness of "t" grid cells (cm)
c     dzw(0) = zT(1)
c     stf     = surface heat flux (cal/cm2/sec)
c     Note: units: result(i,j,k,l)
c        stf = Heat Flux/rho/Cp =(cal/cm2/sec)/(1 g/cm3)/(1. cal/g/oC)
c            = oC-cm/sec

                    IF (iZ.EQ.1) THEN  ! Top term is wind stress term
                      result(i,j,k,l) = (   arg_5(i5,j5,k5,l5)  ! arg_5 = HFLX
     .                                  -  diff_fb(i1,j1,k1,l1,iZ)  )
     .                /(SNGL(zT(1)+zT(2))/2.*Meters_to_cm)
     .                         * sec_per_month 
                    ELSE
                       result(i,j,k,l) = ( diff_fb(i1,j1,k1-1,l1,iZ-1)
     .                                  -  diff_fb(i1,j1,k1,l1,iZ) )
     .                /(SNGL(zT(iZ+1)-zT(iZ-1))/2. * Meters_to_cm)
     .                         * sec_per_month 
                    ENDIF


!          GOTO 1009 
          PRINT*
          PRINT*
          PRINT*,'i,j,k,l     = ', i,j,k,l
          PRINT*,'i1,j1,k1,l1 = ', i1,j1,k1,l1
          PRINT*,'i2,j2,k2,l2 = ', i2,j2,k2,l2
          PRINT*,'i3,j3,k3,l3 = ', i3,j3,k3,l3
          PRINT*,'i4,j4,k4,l4 = ', i4,j4,k4,l4
          PRINT*,'i5,j5,k5,l5 = ', i5,j5,k5,l5
          PRINT*,'iZ             = ',iZ
          PRINT*,'zT(iZ)         = ',zT(iZ)
          PRINT*,'get_dzw(iZ,zT) = ',get_dzw(iZ,zT)
          PRINT*,'T(i3,j3,k3  ,l3) = ', arg_3(i3,j3,k3  ,l3)
          PRINT*,'S(i4,j4,k4  ,l4) = ', arg_4(i4,j4,k4  ,l4)
          PRINT*,'T(i3,j3,k3+1,l3) = ', arg_3(i3,j3,k3+1,l3)
          PRINT*,'S(i4,j4,k4+1,l4) = ', arg_4(i4,j4,k4+1,l4) 
          PRINT*,'T(i3+1,j3,k3  ,l3) = ', arg_3(i3+1,j3,k3  ,l3)
          PRINT*,'S(i4+1,j4,k4  ,l4) = ', arg_4(i4+1,j4,k4  ,l4)
          PRINT*,'T(i3+1,j3,k3+1,l3) = ', arg_3(i3+1,j3,k3+1,l3)
          PRINT*,'S(i4+1,j4,k4+1,l4) = ', arg_4(i4+1,j4,k4+1,l4) 
          PRINT*,'T(i3,j3+1,k3  ,l3) = ', arg_3(i3,j3+1,k3  ,l3)
          PRINT*,'S(i4,j4+1,k4  ,l4) = ', arg_4(i4,j4+1,k4  ,l4)
          PRINT*,'T(i3,j3+1,k3+1,l3) = ', arg_3(i3,j3+1,k3+1,l3)
          PRINT*,'S(i4,j4+1,k4+1,l4) = ', arg_4(i4,j4+1,k4+1,l4) 
          PRINT*,'T(i3+1,j3+1,k3  ,l3) = ', arg_3(i3+1,j3+1,k3  ,l3)
          PRINT*,'S(i4+1,j4+1,k4  ,l4) = ', arg_4(i4+1,j4+1,k4  ,l4)
          PRINT*,'T(i3+1,j3+1,k3+1,l3) = ', arg_3(i3+1,j3+1,k3+1,l3)
          PRINT*,'S(i4+1,j4+1,k4+1,l4) = ', arg_4(i4+1,j4+1,k4+1,l4) 
          PRINT*,'Potential_density(1) = ',
     .           Potential_density(arg_3(i3,j3,k3  ,l3),   ! arg_3 = T
     .                             arg_4(i4,j4,k4  ,l4))   ! arg_4 = S
          PRINT*,'Potential_density(2) = ',
     .           Potential_density(arg_3(i3,j3,k3+1,l3),
     .                             arg_4(i4,j4,k4+1,l4)) 
          PRINT*,'rhom1z(1) = ',rhom1z(i3  ,j3+1,k3,l3,i4  ,j4+1,k4,l4)  
          PRINT*,'rhom1z(2) = ',rhom1z(i3+1,j3+1,k3,l3,i4+1,j4+1,k4,l4) 
          PRINT*,'rhom1z(3) = ',rhom1z(i3  ,j3  ,k3,l3,i4  ,j4  ,k4,l4)  
          PRINT*,'rhom1z(4) = ',rhom1z(i3+1,j3  ,k3,l3,i4+1,j4  ,k4,l4) 
          PRINT*,'Riu(1) = ',Riu(i1, j1, k1,l1, i2, j2, k2,l2,
     .                         i3,  j3,  k3,l3, i4,  j4,  k4,l4,iZ)
          PRINT*,'Riu(2) = ',Riu(i1-1,j1,  k1,l1, i2-1,j2,  k2,l2,
     .                         i3-1,j3,  k3,l3, i4-1,j4,  k4,l4,iZ)
          PRINT*,'Riu(3) = ',Riu(i1,  j1-1,k1,l1, i2,  j2-1,k2,l2,
     .                         i3,  j3-1,k3,l3, i4,  j4-1,k4,l4,iZ)
          PRINT*,'Riu(4) = ',Riu(i1-1,j1-1,k1,l1, i2-1,j2-1,k2,l2,
     .                         i3-1,j3-1,k3,l3, i4-1,j4-1,k4,l4,iZ) 
          PRINT*,'Ri     = ',Ri
          PRINT*,'HFLX   = ',arg_5(i5,j5,k5,l5)
          PRINT*,'diff_cbt = ',diff_cbt
          PRINT*,'diff_fb  = ',diff_fb(i1,j1,k1,l1,iZ)
          PRINT*,'result   = ',result(i,j,k,l)
          PRINT*,'SNGL(zT(1)+zT(2))/2.*Meters_to_cm',
     .           SNGL(zT(1)+zT(2))/2.*Meters_to_cm
          PRINT* 
          PRINT*,'*******************************************'
          PRINT*
 1009     CONTINUE

                  ENDIF
                  iZ = iZ + 1 
                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
                  k3 = k3 + arg_incr(Z_AXIS,ARG3)
                  k4 = k4 + arg_incr(Z_AXIS,ARG4)
                  k5 = k5 + arg_incr(Z_AXIS,ARG5)
               ENDDO
               i1 = i1 + arg_incr(X_AXIS,ARG1)
               i2 = i2 + arg_incr(X_AXIS,ARG2)
               i3 = i3 + arg_incr(X_AXIS,ARG3)
               i4 = i4 + arg_incr(X_AXIS,ARG4)
               i5 = i5 + arg_incr(X_AXIS,ARG5)
            ENDDO
            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
            j5 = j5 + arg_incr(Y_AXIS,ARG5)
         ENDDO               
         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         l5 = l5 + arg_incr(T_AXIS,ARG5)      
      ENDDO
      END




      REAL FUNCTION Potential_density(T,S) ! Density at one Bar 
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
      INCLUDE 'rr_parameters.h'
      INTEGER iZ
      REAL*8 zT(iZ+1)
      IF(iZ.EQ.0)THEN
	get_dzw=SNGL(zT(1))*Meters_to_cm 
      ELSE
        get_dzw=SNGL(zT(iZ+1)-zT(iZ))*Meters_to_cm
      ENDIF
      END


