*
* nco.F
*
* Ansley Manke
* March 1, 2005
*
* This function spawns any NCO call, with the first argument being the
* NCO function name, and the second one long string which is the argument 
* to nco including file names and actions.
*
* 
      SUBROUTINE nco_init(id)

* Define arguments and result

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Call an NCO utility' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, ABSTRACT, NORMAL, NORMAL, 
     .                             NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'operator')
      CALL ef_set_arg_desc(id, arg, 'name of operator')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'arguments')
      CALL ef_set_arg_desc(id, arg, 'one string with all arguments')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      RETURN 
      END

* Define abstract output axis: 1 value

      SUBROUTINE nco_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

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

      INTEGER STRINGLEN, arg, blen, slen

      CHARACTER*2048 argstring, buff

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      blen = 0

      arg = 1
      CALL ef_get_arg_string(id, arg, argstring)

      slen = STRINGLEN(argstring)
      buff = argstring(1:slen)//' -O -h '
      blen = slen + 7

      arg = 2
      CALL ef_get_arg_string(id, arg, argstring)

      slen = STRINGLEN(argstring)
      buff = buff(1:blen)//argstring(1:slen)
      blen = blen + slen 

* Spawn the command 

      result(res_lo_ss(X_AXIS), res_lo_ss(Y_AXIS), res_lo_ss(Z_AXIS), 
     .         res_lo_ss(T_AXIS)) = bad_flag_result

      CALL system( buff(1:blen) )

      result(res_lo_ss(X_AXIS), res_lo_ss(Y_AXIS), res_lo_ss(Z_AXIS), 
     .         res_lo_ss(T_AXIS)) = 1

      RETURN 


      END

        INTEGER FUNCTION STRINGLEN (strin)

* determine the length of a character string as the position of the last 
* non-blank character 

* from tm_lenstr (use tm_lenstr if this is ever converted to an
* internally-linked function)

* calling argument declarations:
	CHARACTER*(*) strin

* internal variable declarations:
	INTEGER	mright, i

* initialize: find highest possible right hand limit of string
	mright = LEN(strin)

        DO 100 i = mright,1,-1	
           IF (strin(i:i) .NE. ' ') GOTO 200	 
  100   CONTINUE

* for all blanks
        STRINGLEN = 0

* Found non-blank
  200   STRINGLEN = i

	RETURN
	END
