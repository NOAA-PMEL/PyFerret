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
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'P')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Confidence Limit')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'DF')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Degrees of freedom')
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
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1, p, df
      REAL GetStudentT

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF ( arg_1(i,j,k,l) .EQ. bad_flag(1) .OR.
     .                 arg_2(i,j,k,l) .EQ. bad_flag(2)) THEN

                     result(i,j,k,l) = bad_flag_result

                  ELSE

                     p = arg_1(i,j,k,l)
                     df = arg_2(i,j,k,l)

                     IF (P.NE.90 .AND. P.NE.95 .AND. P.NE.99)
     .                   CALL EF_BAIL_OUT(id,'P must be 90,95 or 99')
                     IF (df .LE. 0)
     .                   CALL EF_BAIL_OUT(id,'DF must be positive')

                     result(i,j,k,l) = GetStudentT(p, df)

                  END IF

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
      
      RETURN 
      END

	!******************************************
	! 
	! Get the upper cutoff point of the Student T distribution 
	! (P.341 Koopmans .The Spectral Analysis of Time Series)
	!
	!******************************************
	   
c	REAL FUNCTION GetStudentT(P,Nf)
	REAL FUNCTION GetStudentT(P,df)
        IMPLICIT NONE

        INTEGER Nf,P,df  
 
	REAL ST_90(15)
	REAL ST_95(15)
	REAL ST_99(15)
	REAL T
	DATA ST_90/2.92,2.132,1.943,1.86,1.812,1.782,1.761,1.746,
     .           1.734,1.725,1.717,1.711,1.706,1.701,1.697/
	DATA ST_95/4.303,2.776,2.447,2.306,2.228,2.179,2.145,2.12,
     .           2.101,2.086,2.074,2.064,2.056,2.048,2.042/
  	DATA ST_99/9.925,4.604,3.707,3.355,3.169,3.055,2.977,2.921,
     .           2.878,2.845,2.819,2.797,2.779,2.763,2.75/


c        df=Nf*2
        nf = df/2
        IF (df .EQ. 1) nf = 1

	IF(df.GT.120)THEN
     	  IF(P.EQ.90)THEN
		T=1.645
          ELSEIF(P.EQ.95)THEN
		T=1.96
	  ELSE  !  P.EQ.99
		T=2.576
	  ENDIF
	ELSEIF(df.GT.80)THEN
	  IF(P.EQ.90)THEN
		T=1.658
          ELSEIF(P.EQ.95)THEN
		T=1.98
	  ELSE  !  P.EQ.99
		T=2.617
	  ENDIF
	ELSEIF(df.GT.50)THEN
	  IF(P.EQ.90)THEN
		T=1.671
          ELSEIF(P.EQ.95)THEN
		T=2.
	  ELSE  !  P.EQ.99
		T=2.66
	  ENDIF
	ELSEIF(df.GT.30)THEN
	  IF(P.EQ.90)THEN
		T=1.684
          ELSEIF(P.EQ.95)THEN
		T=2.021
	  ELSE  !  P.EQ.99
		T=2.704
	  ENDIF
	ELSE ! df <= 30
	  IF(P.EQ.90)THEN
	    T=ST_90(Nf)
          ELSEIF(P.EQ.95)THEN
	    T=ST_95(Nf)
	  ELSE  !  P.EQ.99
	    T=ST_99(Nf)
	  ENDIF
	ENDIF
	GetStudentT=T
	END

