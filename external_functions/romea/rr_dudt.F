*           rr_dudt.F
*
*           Rick Romea
*           Jan. 5, 2000
*
*     (Backward) time derivative of u
*     Equation :  du/dt = [ u(t) - u(t-1) ] / dt
*     Units :     cm/s/s
*     MOM2 Grid:  U   
*
******************************************************************

      SUBROUTINE RR_dudt_init(id)
	IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id, arg
      arg = 1

      CALL ef_set_desc             (id,
     .'du/dt (Backward) time derivative (cm/s^2);MOM2 U Grid')
      CALL ef_set_num_args         (id, 1)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok     (id, YES, YES, YES, NO)
      CALL ef_set_arg_name         (id, arg, 'U')
      CALL ef_set_arg_desc         (id, arg,
     . 'Zonal velocity, on the MOM2 U Grid. ')
      CALL ef_set_arg_unit         (id, arg, 'cm/sec')
      CALL ef_set_arg_type         (id, arg, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, arg, YES, YES, YES, YES)

      !  Extend the time axis one point backward

      CALL ef_set_axis_extend      (id, arg, T_AXIS,-1,0)
      END


      SUBROUTINE RR_dudt_compute(id, arg_1, result)
	IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER        id

      REAL           bad_flag(EF_MAX_ARGS)
      REAL           bad_flag_result
      REAL           arg_1 (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .                      mem1loz:mem1hiz, mem1lot:mem1hit)
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

      REAL*8         RR_times(1024)
      LOGICAL        RR_badDt
      REAL           RR_dt
      REAL           RR_factor
      INTEGER        iT

      CHARACTER *16  ax_name(4)
      CHARACTER *16  ax_units(4)
      LOGICAL        backward(4)
      LOGICAL        modulo(4)
      LOGICAL        regular(4)

      INCLUDE 'rr_parameters.h'

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)
      CALL ef_get_coordinates    (id, ARG1, T_AXIS, 
     .                            arg_lo_ss(T_AXIS,ARG1),
     .                            arg_hi_ss(T_AXIS,ARG1), RR_times)

      CALL ef_get_axis_info(id, ARG1, ax_name, ax_units, 
     .                                backward, modulo, regular) 

      IF (ax_units(4) .EQ. 'sec' ) THEN 
         RR_factor = 1.
      ELSEIF (ax_units(4) .EQ. 'min')THEN
         RR_factor = 60.   
      ELSEIF (ax_units(4) .EQ. 'hour' .OR. 
     .        ax_units(4) .EQ. 'hours' )THEN
         RR_factor = 3600.  
      ELSEIF (ax_units(4) .EQ. 'day' .OR.
     .        ax_units(4) .EQ. 'days'  )THEN
         RR_factor = 86400. 
      ELSE
        WRITE (6,*)'Time axis units =', ax_units(4) 
       !CALL ef_bail_out(id,
       !.        'Time axis not specified as  sec, min, hour,or day.')
      ENDIF

      iT = 2
      l1 = arg_lo_ss(T_AXIS,ARG1) + 1     ! Note: extended time axis
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         
         RR_dt =  SNGL(RR_times(iT) - RR_times(iT-1))
         RR_badDt = RR_dt .EQ. 0.
 
         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1)
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  IF (arg_1(i1,j1,k1,l1)   .EQ. bad_flag(ARG1) .OR.
     .                arg_1(i1,j1,k1,l1-1) .EQ. bad_flag(ARG1) .OR.
     .                RR_badDt) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE
                     result(i,j,k,l) = ( arg_1(i1,j1,k1,l1)
     .                    -   arg_1(i1,j1,k1,l1-1) ) / RR_dt
     .                    / RR_factor	* sec_per_month
                  ENDIF
               
                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
               ENDDO
               i1 = i1 + arg_incr(X_AXIS,ARG1)
            ENDDO

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
         ENDDO

	   iT = iT+1
         l1 = l1 + arg_incr(T_AXIS,ARG1)
      ENDDO

      END











