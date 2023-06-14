*
* efstrings.F
*
*
* Returns A testing additional string capabilities
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, efstrings )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE efstrings_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'Demo Function: tests string utilities' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE efstrings_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work array X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER iarg, ns
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      iarg = 1
      ns = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1

      CALL ef_set_work_array_dims(id, iarg, 1,1,1,1, 1,1,1,2*ns)

      iarg = 2
      CALL ef_set_work_array_dims(id, iarg, 1,1,1,1, 1,1,1,20*ns)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* In this subroutine we compute the result
*
      SUBROUTINE efstrings_compute(id, arg_1, result, tax)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit)
      REAL*8 tax(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .           wrk1loz:wrk1hiz, wrk1lot:wrk1hit/2)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER rtype, iarg, slen, prec
      CHARACTER*32 textstring, datestring
      CHARACTER*3 short

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_result_type(id, rtype)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates (id, ARG1, T_AXIS, 
     .       arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), tax)

*     'dd-mon-yyyy hh:mm:ss' date format
      prec = 6

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)


         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

C get the string and its length
                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1, slen, textstring)

C get just the length
                  CALL EF_GET_STRING_ARG_ELEMENT_LEN(id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1,  slen)

C What if the string buffer were declared too short?
                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1, slen, short)

C just copy the pointer to the result
                  IF (l .EQ. res_lo_ss(T_AXIS)) THEN
                     CALL EF_PUT_STRING_PTR
     .                 (arg_1(1,i1,j1,k1,l1),result(1,i,j,k,l))

C Copy a new string to the result
                  ELSE IF (l .EQ. res_lo_ss(T_AXIS)+1) THEN
                     textstring = 'new'
                     slen = 3
                     CALL EF_PUT_STRING (textstring, slen, result(1,i,j,k,l))
                  ELSE

C Compute other string values and put them in the result.
                     CALL EF_GET_AXIS_DATES (id, ARG1, tax(1,1,1,L1), 1, prec, slen, datestring)
                     CALL EF_PUT_STRING (datestring, slen, result(1,i,j,k,l))
                  ENDIF

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
     
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
