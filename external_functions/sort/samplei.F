*
* samplei.F
*
* Ansley Manke
* Dec 1998
*
* This function samples data along the I axis using a set of indices
*
*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
* 
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE samplei_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to a set of I indices')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'I_INDICES')
      CALL ef_set_arg_desc(id, arg, 'indices upon which to sample')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using I indices')
      CALL ef_set_axis_influence(id, arg, NO, YES, YES, YES)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE samplei_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
     .     
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .      memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      INTEGER id, i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER j2, k2, l2
      INTEGER iorder
      REAL aorder
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      DO 400 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS) 

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 200 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                i1 = arg_lo_ss(X_AXIS,ARG1)

                DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                   aorder = arg_1(i1,j1,k1,l1)
                   iorder = aorder

*  Check that we do have an index of ARG_2; set the result

                  IF (aorder .EQ. bad_flag(ARG1)  .OR. 
     .                arg_2(iorder,j2,k2,l2) .EQ. bad_flag(ARG2))
     .              THEN
                        result(i,j,k,l) = bad_flag_result

                   ELSE
                      IF (iorder .LT. arg_lo_ss(X_AXIS,ARG2)  .OR.
     .                    iorder .GT. arg_hi_ss(X_AXIS,ARG2) ) THEN
                        GO TO 999

                      ELSE
                        result(i,j,k,l) = arg_2(iorder,j2,k2,l2)

                     END IF
                  END IF

                  i1 = i1 + arg_incr(X_AXIS,ARG1)
 100           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
               l2 = l2 + arg_incr(T_AXIS,ARG2)
 200        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
            k2 = k2 + arg_incr(Z_AXIS,ARG2)
 300     CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
 400  CONTINUE

      RETURN 

999   CONTINUE

      WRITE (errtxt,10) iorder, arg_lo_ss(X_AXIS,ARG2), 
     .              arg_hi_ss(X_AXIS,ARG2)

   10 FORMAT ('Sample index ', i8, ' in ARG1 outside the range ', 
     .        'of indices for ARG2 ('i8,':',i8')')

      CALL EF_BAIL_OUT(id, errtxt)

      RETURN
      END
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************
