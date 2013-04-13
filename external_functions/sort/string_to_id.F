*
* string_to_id.F
*
* Ansley Manke
* March 2013
*
*  Convert a sorted list of strings to a list of numeric ID's:
*    the first string encountered is ID=1,
*    the next unique string encountered is ID=2, etc.
*
* This function takes two string arguments and one numeric argument.
* It returns a string result.  The result is the first N letters of the 
* first argument prepended to the string in the second argument, where N
* is the number of characters to take from the strings of arg1.
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


      SUBROUTINE string_to_id_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Return numeric IDs for each new string in sorted list of strings')

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     . IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'STRINGS')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'string argument')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE string_to_id_compute(id, arg_1,  result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result

*  ! string arg and result need to be twice the length.

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l, i1,i2,i3,i4
      INTEGER iarg, slen1, alen, nchar
      REAL value
      CHARACTER text1*1000, this_string*1000
      LOGICAL isnew

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      aindex = 0
      alen = 1
      this_string = " "

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
                  iarg = 1
 	          CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1, slen1, text1)
                  CALL EF_GET_STRING_ARG_ELEMENT_LEN (id, iarg, arg_1, 
     .                                           i1,j1,k1,l1, slen1)

                  isnew = .FALSE.
                  IF (slen1 .EQ. alen) THEN
	             IF (text1(:slen1) .NE. this_string(:alen)) isnew = .TRUE.
                  ELSE
	             isnew = .TRUE.
                  ENDIF

                  IF (isnew) THEN
	             aindex = aindex + 1
	             this_string = text1
	             alen = slen1
                  ENDIF
	    
                  result(i,j,k,l) = aindex

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE
 

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
