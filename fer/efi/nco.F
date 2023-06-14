*
* nco.F
*
* Ansley Manke
* March 1, 2005
* V685 *acm* 5/13  If the user has sent -A (append) then don't send 
*                  -O (override) to nco.
*
* This function spawns any NCO call, with the first argument being the
* NCO function name, and the second one long string which is the argument 
* to nco including file names and actions.
*
* 
      SUBROUTINE nco_init(id)

* Define arguments and result

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Call an NCO utility' )
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                NORMAL,   NORMAL,
     .                                NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'operator')
      CALL ef_set_arg_desc(id, arg, 'name of operator')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'arguments')
      CALL ef_set_arg_desc(id, arg, 'one string with all arguments')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      RETURN 
      END

* Define abstract output axis: 1 value

      SUBROUTINE nco_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER ivalue

      ivalue = 1
      CALL ef_set_axis_limits(id, X_AXIS, ivalue, ivalue)

      RETURN 
      END

*
* Compute the result
*
      SUBROUTINE nco_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

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

      LOGICAL TM_HAS_STRING
      INTEGER TM_LENSTR

      INTEGER arg, alen, slen
      INTEGER i, j, k, l, m, n
      CHARACTER*2048 argstring, buff

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      arg = 1
      CALL ef_get_arg_string(id, arg, argstring)
      alen = TM_LENSTR(argstring)
      buff = argstring(1:alen)
      slen = alen

      arg = 2
      CALL ef_get_arg_string(id, arg, argstring)
      alen = TM_LENSTR(argstring)

      IF ( TM_HAS_STRING(argstring, '-A') ) THEN 
         buff = buff(1:slen)//' -h '
         slen = slen + 5
      ELSE
         buff = buff(1:slen)//' -O -h '
         slen = slen + 7
      ENDIF

      buff = buff(1:slen)//argstring(1:alen)
      slen = slen + alen 

* Spawn the command 

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)
      result(i,j,k,l,m,n) = bad_flag_result

      CALL system( buff(1:slen) )

      result(i,j,k,l,m,n) = 1

      RETURN 
      END
