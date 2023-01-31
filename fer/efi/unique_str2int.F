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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 descrip

      WRITE (descrip, 100)
  100 FORMAT ('Returns an ID number for each unique string ',
     .        'in the input array of strings (case sensitive)')
      CALL ef_set_desc(id, descrip)

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Array of Strings')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we compute the result
*
      SUBROUTINE unique_str2int_compute(id, arg_1, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit, 
     .           mem1loe:mem1hie, mem1lof:mem1hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

* Routine US2I_COMPARE_STRING_LIST creates and maintains a 
* list of unique names and their sequence numbers.
*
* Future calls to this function will return the same indices for strings
* given in this call because the above functions maintains a static list
* of strings which is initialized once and never released.

      INTEGER flen
      PARAMETER (flen=512)

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER b1len, status, isame, icmp, ifirst, istr
      CHARACTER*512 buff1
      REAL result_last
      INTEGER*1 fhol(flen), hol_last(flen)      ! c-type Hollerith string buffer

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      ifirst = 1

* These loops are intentionally left in un-optimal order since
* contiguous identical strings are most likely at a location
* through all time/ensemble/forecast periods.

      i = res_lo_ss(X_AXIS)
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

      j = res_lo_ss(Y_AXIS)
      DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)

      k = res_lo_ss(Z_AXIS)
      DO 300 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)

      l = res_lo_ss(T_AXIS)
      DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

      m = res_lo_ss(E_AXIS)
      DO 500 m1 = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)

      n = res_lo_ss(F_AXIS)
      DO 600 n1 = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)

         CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG1, arg_1,
     .                          i1, j1, k1, l1, m1, n1, b1len, buff1)

         CALL TM_FTOC_STRNG(buff1(1:b1len), fhol, flen)
         isame = 0
         IF ( ifirst .EQ. 0 ) THEN
*           hol_last is assigned from the last search, so first
*           check the common case of contiguous identical strings
            CALL US2I_STR_CMP(fhol, hol_last, icmp)
            IF ( icmp .EQ. 0 ) THEN
               result(i,j,k,l,m,n) = result_last
               isame = 1
            ENDIF
         ENDIF
         IF ( isame .EQ. 0 ) THEN
*           Search for the string in the whole list
*           (initializing the list if needed)
            CALL US2I_COMPARE_STRING_LIST(fhol, istr)
            IF ( istr .EQ. 0 ) THEN
               buff1 = 'Problems initializing the list of strings'
               GOTO 999
            ENDIF
            result(i,j,k,l,m,n) = FLOAT(istr)
*           Copy the string and the result for the next compare
            DO istr = 1, b1len
               hol_last(istr) = fhol(istr)
            ENDDO
            result_last = result(i,j,k,l,m,n)
            ifirst = 0
         ENDIF

         n = n + res_incr(F_AXIS)
 600  CONTINUE

         m = m + res_incr(E_AXIS)
 500  CONTINUE

         l = l + res_incr(T_AXIS)
 400  CONTINUE

         k = k + res_incr(Z_AXIS)
 300  CONTINUE

         j = j + res_incr(Y_AXIS)
 200  CONTINUE

         i = i + res_incr(X_AXIS)
 100  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, buff1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
