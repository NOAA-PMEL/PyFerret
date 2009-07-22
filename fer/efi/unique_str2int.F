*
* unique_str2int.F
*
* Ansley Manke
* July 2009
*
* This function returns an ID number for each unique
* string in the input array of strings.
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


      SUBROUTINE unique_str2int_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg
      CHARACTER*100 descrip

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      WRITE (descrip, 100)
      CALL ef_set_desc(id, descrip)
  100 FORMAT ('Returns an ID number for each unique string',
     .        'in the input array of strings (case sensitive)')

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_desc(id, arg, 'Array of Strings')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we compute the result
*
      SUBROUTINE unique_str2int_compute(id, arg_1, result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(2,mem1lox:mem1hix, mem1loy:mem1hiy, 
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

      INTEGER INIT_US2I_LIST, US2I_COMPARE_STRING_LIST
      INTEGER i,j,k,l, i1,j1,k1,l1, flen, b1len, status, isame, 
     .        icmp, ifirst
      CHARACTER*512 buff1
      REAL result_last

      PARAMETER (flen=512)
      INTEGER*1 fhol(flen), ghol(flen)      ! c-type Hollerith string buffer


* Routine US2I_COMPARE_STRING_LIST creates and maintains a 
* list of unique names and their sequence numbers.

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      ifirst = 1

      i = 1   ! res_lo_ss(X_AXIS)
      DO 400 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

         j = res_lo_ss(Y_AXIS)
         DO 300 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)

            k = res_lo_ss(Z_AXIS)
            DO 200 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)

               l = res_lo_ss(T_AXIS)
               DO 100 l1 = arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)

                   CALL EF_GET_STRING_ARG_ELEMENT(id, ARG1, arg_1,
     .                       i1,j1,k1,l1, b1len, buff1)
                  
                   isame = 0
		   IF ( ifirst .EQ. 1) THEN
                      CALL TM_FTOC_STRNG( buff1(1:b1len), fhol, flen )
		      status = INIT_US2I_LIST(fhol)
		      result(i,j,k,l) = 1.
		      ifirst = 0
		   ELSE
		      CALL TM_FTOC_STRNG( buff1(1:b1len), fhol, flen )
		      CALL US2I_STR_CMP(fhol, ghol, icmp)
		      IF (icmp .EQ. 0) THEN
		         result(i,j,k,l) = result_last
		         isame = 1
		      ELSE
		         istr = 0
		         status = US2I_COMPARE_STRING_LIST (fhol, istr)
		         result(i,j,k,l) = FLOAT(istr)
		      ENDIF
		   ENDIF

                   IF (isame .NE. 1) THEN
                      DO m = 1, b1len
		         ghol(m) = fhol(m)
		      ENDDO
		   ENDIF
		   result_last = result(i,j,k,l)

                  l = l + res_incr(T_AXIS)
 100           CONTINUE

               k = k + res_incr(Z_AXIS)
 200        CONTINUE

            j = j + res_incr(Y_AXIS)
 300     CONTINUE

         i = i + res_incr(X_AXIS)
 400  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
