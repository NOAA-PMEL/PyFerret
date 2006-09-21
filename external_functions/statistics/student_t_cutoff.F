*
* student_t_cutoff.F
*
* Ansley Manke
* April 4, 2005
*

* This function returns the upper cutoff point of the Student T distribution 
*  (P.341 Koopmans .The Spectral Analysis of Time Series)
*  From Rick Romea's student_t.F


      SUBROUTINE student_t_cutoff_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Return student-t cutoff' )

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .    IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'P')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Confidence Limit')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'nf')
      CALL ef_set_arg_unit(id, arg, 'Number in sample')

      CALL ef_set_arg_desc(id, arg, 'nf')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE student_t_cutoff_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

	  INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .          mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .          mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .           memresloz:memreshiz, memreslot:memreshit)

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .    arg_incr(4,1:EF_MAX_ARGS)

      INTEGER arg
      INTEGER i,j,k,l
      INTEGER i2, j2, k2, l2, p, nf
      REAL value, GetStudentT

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C test value of P 

      arg = 1
      CALL ef_get_one_val(id, arg, value)
      p = value

      IF (P.NE.90 .AND. P.NE.95 .AND. P.NE.99)
     .         CALL EF_BAIL_OUT(id,'P must be 90, 95 or 99')

C Loop over field of N, return value of missing if n is not at least 1

      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k2 = arg_lo_ss(Z_AXIS,ARG2)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l2 = arg_lo_ss(T_AXIS,ARG2)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF ( arg_2(i2,j2,k2,l2) .EQ. bad_flag(2)) THEN
                     result(i,j,k,l) = bad_flag_result

                  ELSE
                     nf = arg_2(i2,j2,k2,l2)
                     IF ( nf .LE. 1) THEN
                        result(i,j,k,l) = bad_flag_result
                     ELSE
                        result(i,j,k,l) = GetStudentT(p, nf)
                     ENDIF

                  END IF

                  l2 = l2 + arg_incr(T_AXIS,ARG2)
 100           CONTINUE

               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE
      
      RETURN 
      END

       REAL FUNCTION GetStudentT(P,Nf)
       IMPLICIT NONE
        INTEGER Nf,P,df    
        REAL ST_90(100)
        REAL ST_95(100)
        REAL ST_99(100)
        REAL T
        DATA ST_90/6.314,2.920,2.353,2.132,2.015,1.943,1.895,1.860,
     .          1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,
     .          1.740,1.734,1.729,1.725,1.721,1.717,1.714,1.711,
     .          1.708,1.706,1.703,1.701,1.699,1.697,1.696,1.694,
     .          1.692,1.691,1.690,1.688,1.687,1.686,1.685,1.684,
     .          1.683,1.682,1.681,1.680,1.679,1.679,1.678,1.677,
     .          1.677,1.676,1.675,1.675,1.674,1.674,1.673,1.673,
     .          1.672,1.672,1.671,1.671,1.670,1.670,1.669,1.669,
     .          1.669,1.668,1.668,1.668,1.667,1.667,1.667,1.666,
     .          1.666,1.666,1.665,1.665,1.665,1.665,1.664,1.664,
     .          1.664,1.664,1.663,1.663,1.663,1.663,1.663,1.662,
     .          1.662,1.662,1.662,1.662,1.661,1.661,1.661,1.661,
     .          1.661,1.661,1.660,1.660/
      DATA ST_95/12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,
     .          2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,
     .          2.093,2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,
     .          2.048,2.045,2.042,2.040,2.037,2.035,2.032,2.030,2.028,
     .          2.026,2.024,2.023,2.021,2.020,2.018,2.017,2.015,2.014,
     .          2.013,2.012,2.011,2.010,2.009,2.008,2.007,2.006,2.005,
     .          2.004,2.003,2.002,2.002,2.001,2.000,2.000,1.999,1.998,
     .          1.998,1.997,1.997,1.996,1.995,1.995,1.994,1.994,1.993,
     .          1.993,1.993,1.992,1.992,1.991,1.991,1.990,1.990,1.990,
     .          1.989,1.989,1.989,1.988,1.988,1.988,1.987,1.987,1.987,
     .          1.986,1.986,1.986,1.986,1.985,1.985,1.985,1.984,1.984,
     .          1.984/
      DATA ST_99/63.657,9.925,5.841,4.604,4.032,3.707,3.499,3.355,3.250,
     .          3.169,3.106,3.055,3.012,2.977,2.947,2.921,2.898,2.878,
     .          2.861,2.845,2.831,2.819,2.807,2.797,2.787,2.779,2.771,
     .          2.763,2.756,2.750,2.744,2.738,2.733,2.728,2.724,2.719,
     .          2.715,2.712,2.708,2.704,2.701,2.698,2.695,2.692,2.690,
     .          2.687,2.685,2.682,2.680,2.678,2.676,2.674,2.672,2.670,
     .          2.668,2.667,2.665,2.663,2.662,2.660,2.659,2.657,2.656,
     .          2.655,2.654,2.652,2.651,2.650,2.649,2.648,2.647,2.646,
     .          2.645,2.644,2.643,2.642,2.641,2.640,2.640,2.639,2.638,
     .          2.637,2.636,2.636,2.635,2.634,2.634,2.633,2.632,2.632,
     .          2.631,2.630,2.630,2.629,2.629,2.628,2.627,2.627,2.626,
     .          2.626/

        IF(P.NE.90.AND.P.NE.95.AND.P.NE.99)THEN
               P=90
        ENDIF

        df=Nf-1
        IF (df .LE. 0) then
           t =1.
        endif

        IF(df.GT.100)THEN
               IF(P.EQ.90)THEN
            T=1.645
              ELSEIF(P.EQ.95)THEN
            T=1.96
          ELSE  !  P.EQ.99
            T=2.576
          ENDIF
        ELSE ! df <= 100
          IF(P.EQ.90)THEN
            T=ST_90(df)
              ELSEIF(P.EQ.95)THEN
            T=ST_95(df)
          ELSE  !  P.EQ.99
            T=ST_99(df)
          ENDIF
        ENDIF
        GetStudentT=T
        END
