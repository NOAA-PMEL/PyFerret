*
* compressj.F
*
* Ansley Manke
* August 1998
*
* This function compresses data on J axis, ending with missing data.
*
*  NOTE:
*  IT IS GENERALLY ADVISABLE TO INCLUDE EXPLICIT LIMITS WHEN WORKING WITH
*  FUNCTIONS THAT REPLACE AXES. FOR EXAMPLE, THE CONSIDER THE FUNCTION
*  COMPRESSL(v). THE EXPRESSION
*  	LIST/L=6:10 COMPRESSL(v)
*  IS NOT EQUIVALENT TO
*  	LIST COMPRESSL(v[L=6:10])
*  THE FORMER WILL LIST THE 6TH THROUGH 10TH COMPRESSED INDICES FROM THE ENTIRE
*  L RANGE OF VARIABLE V. THE LATTER WILL LIST ALL OF THE INDICES THAT
*  RESULT FROM COMPRESSING v[L=6:10].
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


      SUBROUTINE compressj_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data, compressed along the J axis: Missing ',
     .        'points moved to the end')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 1)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, ABSTRACT, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT')
      CALL ef_set_arg_desc(id, arg, 'variable to compress in J')
      CALL ef_set_axis_influence(id, arg, YES, NO, YES, YES)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE compressj_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_lo_l = 1
      my_hi_l = 1

      arg = 1

      my_hi_l = arg_hi_ss(Y_AXIS,arg) - arg_lo_ss(Y_AXIS,arg) + 1


      CALL ef_set_axis_limits(id, Y_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE compressj_compute(id, arg_1, result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

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

      INTEGER id
      INTEGER nbad, m_indx
      INTEGER m, nsrt
      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1

*  Get subscript ranges

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      i1 = arg_lo_ss(X_AXIS, ARG1)
      DO 500 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         k1 = arg_lo_ss(Z_AXIS, ARG1)
         DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS, ARG1)
            DO 300 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               nsrt = 0
               nbad = 0

               j1 = arg_lo_ss(Y_AXIS,ARG1)
               DO 100 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                  IF (arg_1(i1,j1,k1,l1) .NE. bad_flag(1)) THEN
                     nsrt = nsrt + 1
                     result(i,nsrt,k,l) = arg_1(i1,j1,k1,l1)

                  ELSE
                     nbad = nbad + 1
                  END IF

                  j1 = j1 + arg_incr(Y_AXIS,ARG1)
 100           CONTINUE
      
*
* Compress argument, putting bad data at the end.
*
               DO 200 m = 1, nbad
                  m_indx = m + nsrt
                  result(i,m_indx,k,l) = bad_flag_result
 200           CONTINUE

               l1 = l1 + arg_incr(T_AXIS, ARG1)
 300        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS, ARG1)
 400     CONTINUE

         i1 = i1 + arg_incr(X_AXIS, ARG1)
 500  CONTINUE
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END
