* chdir.F
*
* Change current working directory.
* -Andrew Wittenberg 10/17/01  Princeton University

      SUBROUTINE chdir_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_set_desc(id,'Change current working directory.' )
      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name(id, arg, 'TO_DIRECTORY')
      CALL ef_set_arg_desc(id, arg, 'desired working directory')
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
      SUBROUTINE chdir_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER itest
      CHARACTER arg1_text*160

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_get_arg_string(id, 1, arg1_text)

      WRITE(*,*) 'Changing directory to: ',trim(arg1_text)
      itest = CHDIR(trim(arg1_text))

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
