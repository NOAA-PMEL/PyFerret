*  rr_geopotential_height.F
*
*     Rick Romea
*     Jan. 20, 2000
*
*  Geopotential height (cm) = 
*      Geopotential meters (m^2/s^2) / gravity (9.8 m/s^2) * 100
*  
************************************************************************

      SUBROUTINE RR_geopotential_height_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id,' Geopotential height (cm)')
      CALL ef_set_num_args         (id, 2)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_axis_reduction   (id, RETAINED,
     .                                  RETAINED,         
     .                                  REDUCED,
     .                                  RETAINED)          
      CALL ef_set_arg_name         (id, ARG1, 'TEMP')
      CALL ef_set_arg_desc         (id, ARG1, 'In Situ Temperature')
      CALL ef_set_arg_unit         (id, ARG1, 'oC')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES,YES,YES,YES)

      CALL ef_set_arg_name         (id, ARG2, 'SALT')
      CALL ef_set_arg_desc         (id, ARG2, 'Salinity')
      CALL ef_set_arg_unit         (id, ARG2, 'psu')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, YES,YES,YES,YES)

      END

c*******************************************************************

      SUBROUTINE RR_geopotential_height_compute(id,arg_1,arg_2,result)
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
      REAL result (memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit)

      INTEGER  res_lo_ss (4)
      INTEGER  res_hi_ss (4)
      INTEGER  res_incr  (4)

      INTEGER  arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER  arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER  arg_incr  (4,EF_MAX_ARGS)

      INTEGER  i,  j,  k,   l
      INTEGER  i1, j1, k1, l1
      INTEGER  i2, j2, k2, l2

      REAL*8   zT(1024)

      INTEGER  iZ

      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      REAL       Rho  ! function for density
      REAL       dz,anomaly
      INCLUDE 'rr_parameters.h'

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)

      ! Check that upper z grid point is the surface z grid point
	
      IF (arg_lo_ss(Z_AXIS,ARG1) .NE. 1) THEN
        CALL ef_bail_out(id,'TEMP:Upper grid point must be 1')
      ENDIF
      IF (arg_lo_ss(Z_AXIS,ARG2) .NE. 1) THEN
        CALL ef_bail_out(id,'SALT:Upper grid point must be 1')
      ENDIF
 
      ! Get Z grid coordinates

      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .                            arg_lo_ss(Z_AXIS,ARG1),
     .                            arg_hi_ss(Z_AXIS,ARG1), zT)

      ! check axis units

      CALL ef_get_axis_info  (id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 
      IF     ( ax_units(3) .NE. 'm'       .AND.
     .         ax_units(3) .NE. 'meters'  .AND.
     .         ax_units(3) .NE. 'M'       .AND.
     .         ax_units(3) .NE. 'METERS'  .AND.
     .         ax_units(3) .NE. 'meters_z' ) THEN
        WRITE (6,*)'Z axis units =', ax_units(3) 
        !CALL ef_bail_out(id,'Z axis must be in meters')
      ENDIF

      l1 = arg_lo_ss(T_AXIS,ARG1)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

        j1 = arg_lo_ss(Y_AXIS,ARG1) 
        j2 = arg_lo_ss(Y_AXIS,ARG2) 
        DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           i1 = arg_lo_ss(X_AXIS,ARG1) 
           i2 = arg_lo_ss(X_AXIS,ARG2) 
           DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
              
              ! This is a dummy axit, since z is reduced
              DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                 result(i,j,k,l) = 0.  ! initialize at the surface
                 iZ = 1
                 k2 = arg_lo_ss(Z_AXIS,ARG2)
                 DO k1 = arg_lo_ss(Z_AXIS,ARG1), 
     .                   arg_hi_ss(Z_AXIS,ARG1),
     .                    arg_incr(Z_AXIS,ARG1)     
                   IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .                 arg_2(i2,j2,k2,l2) .EQ. bad_flag(ARG2) ) THEN
                     result(i,j,k,l) = bad_flag_result
                     GOTO 1 ! Bail from this z-integration
                   ELSE

                     anomaly = 
     .                   1. / Rho( arg_1(i1,j1,k1,l1),
     .                           arg_2(i2,j2,k2,l2), SNGL(zT(iZ))) 
     .                -  1. / Rho(  0.,  35.,  SNGL(zT(iZ)))   
           
                     IF(k1.EQ.1)THEN  ! Top z-grid point
                       dz = (SNGL(zT(iZ+1)) + SNGL(zT(iZ)))   / 2.
                     ELSEIF(k1.EQ.arg_hi_ss(Z_AXIS,ARG1))THEN ! Bottom z-point
                       dz = (SNGL(zT(iZ))   - SNGL(zT(iZ-1))) / 2.    
                     ELSE ! middle gridpoints
                       dz = (SNGL(zT(iZ+1)) - SNGL(zT(iZ-1))) / 2.
                     ENDIF
                     result(i,j,k,l)=result(i,j,k,l)+anomaly*dz*1.e5
                   ENDIF

	           iZ = iZ + 1
                   k2 = k2 + arg_incr(Z_AXIS,ARG2)
                 ENDDO
              ENDDO 
 1	      CONTINUE
              i1 = i1 + arg_incr(X_AXIS,ARG1)
              i2 = i2 + arg_incr(X_AXIS,ARG2)
           ENDDO

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
           j2 = j2 + arg_incr(Y_AXIS,ARG2)
         ENDDO

      l1 = l1 + arg_incr(T_AXIS,ARG1)
      l2 = l2 + arg_incr(T_AXIS,ARG2)
      ENDDO

      END



        REAL FUNCTION RHO(temperature,salinity,pressure)

* The 1980 UNESCO International Equation of State (IES80)
* *sh* 8/5/92 - adapted from C version supplied by John Osborne, PMEL

*       This calculates rho (density kg/m^3) at salinity, 
*	temperature, and pressure.  This will be potential density if 
*	temperature is potential temperature.  The routine uses the high 
*	pressure equation of state from Millero et al. 1980 and the one-
*	atmosphere equation of state from Millero and Poisson 1981 as
*	reported in Gill 1982.  The notation follows Millero et al. 1980
*	and Millero and Poisson 1981.
*	
*	Note: the routine takes p in decibars and converts to bars for
*	the calculations.
*	
*  Gill, A.E., 1982, Atmosphere-Ocean Dynamics, Academic Press, Inc., 662 pp. 
*	
*	Input Units:
*			S: psu	T: deg. C  p: decibars.
*			
*	Output Units:
*			rho: kg/m^3
*			
*	Check values:
*			rho = 999.96675 	for S = 0,  T = 5,  p = 0
*			rho = 1027.675465 	for S = 35, T = 5,  p = 0
*			rho = 1062.538172 	for S = 35, T = 25, p = 10000.

        REAL salinity, temperature, pressure
	REAL S, T, p, rootS
	REAL a, b, c, d, e
	REAL Aw, Bw, Kw
	REAL A2, B2, C2, Kzero, K
	REAL rhoW, rhoZero
	S = salinity
	T = temperature
	p = pressure / 10.
	rootS = SQRT (S)
	rhoW = ((((6.536332e-09 * T - 1.120083e-06) * T + 1.001685e-04)
     .		* T - 9.095290e-03) * T + 6.793952e-02) * T + 999.842594
	A2 = (((5.3875e-09 * T - 8.2467e-07) * T + 7.6438e-05)
     .		* T - 4.0899e-03) * T + 8.24493e-01
	B2 = (-1.6546e-06 * T + 1.0227e-04) * T - 5.72466e-03
	C2 = 4.8314e-04
	rhoZero	= (C2 * S + B2 * rootS + A2) * S + rhoW
	IF ( pressure .EQ. 0.0 ) THEN
	   RHO=rhoZero
	   RETURN
        ENDIF
	a = ((-6.1670e-05*T+1.09987e-02)*T-6.03459e-01)*T+54.6746
	b = (-5.3009e-04 * T + 1.6483e-02) * T + 7.944e-02
	c = (-1.6078e-06 * T - 1.0981e-05) * T + 2.2838e-03
	d = 1.91075e-04
	e = (9.1697e-10 * T + 2.0816e-08) * T - 9.9348e-07
	Aw = ((-5.77905e-07*T+1.16092e-04)*T+1.43713e-03)*T+3.239908
	Bw = (5.2787e-08 * T - 6.12293e-06) * T + 8.50935e-05
	Kw = (((-5.155288e-05 * T + 1.360477e-02) * T - 2.327105)
     .		* T + 148.4206) * T + 19652.21
	Kzero = (b * rootS + a) * S + Kw
	K = ((e * p + d * rootS + c) * S + Bw * p + Aw) * p + Kzero
	RHO = rhoZero / (1.0 - (p / K))
	END






