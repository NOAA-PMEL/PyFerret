*
* time_reformat.F
*
* Jan 2015
*
* Given a time string, translate it to a Ferret-formatted string
*   dd-Mon-YYYY hh:mm:ss
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, time_reformat )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE time_reformat_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns date string(s) reformatted to Ferret time-string format' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     . 'Time string(s) to convert, e.g. 1992-04-23 12:30:01')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* In this subroutine we compute the result
*
      SUBROUTINE time_reformat_compute(id, arg_1, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)



* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1

      INTEGER rtype, slen, status
      CHARACTER*20 indate, outdate
      CHARACTER*128 err_msg
      INTEGER cal_id

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_result_type(id, rtype)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 600 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 500 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 400 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 300 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               m1 = arg_lo_ss(E_AXIS,ARG1)
               DO 200 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

               n1 = arg_lo_ss(F_AXIS,ARG1)
               DO 100 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

C   Reformat the date string and put into the result.

                  CALL ef_get_string_arg_element_6d(id, ARG1, arg_1,
     .                i1,j1,k1,l1,m1,n1,slen,indate)

                 CALL TAX_REFORMAT_DATE (indate, outdate, status)
		 IF (outdate(1:3) .EQ. 'xxx') GOTO 5000

                 slen = 20
                 CALL EF_PUT_STRING (outdate, slen,
     .                    result(i,j,k,l,m,n))

                  n1 = n1 + arg_incr(F_AXIS,ARG1)
 100           CONTINUE
                 
                  m1 = m1 + arg_incr(E_AXIS,ARG1)
 200           CONTINUE
                 
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 300           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 400        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 500     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 600  CONTINUE
     
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
 5000 err_msg = 'Could not translate string to Ferret date format '// indate
      CALL ef_bail_out (id, err_msg)

      END
