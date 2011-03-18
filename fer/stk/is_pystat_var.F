* copy a pystat variable in ferret memory

      SUBROUTINE IS_PYSTAT_VAR(memory, *, *, sts)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'interp_stack.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xpyvar_info.cmn'

* calling argument declarations:
      INTEGER sts
      REAL    memory(mem_blk_size, max_mem_blks)

* internal variable declarations:
      INTEGER cx, uvar, mr

* initialize
      cx = is_cx(isp)
      uvar = cx_variable(cx)

* reserve and catalog space for the result
      CALL CREATE_TEMP_MEM_VAR(cx, mr, sts)
      IF ( sts .NE. ferr_ok ) GOTO 5000

* diagnostic output
      IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('pystat_var', mr, point_to_mr)

      CALL COPY_PYSTAT_DATA(memory(1,mr_blk1(mr)), pyvar_ndarray_obj(uvar))

* set the bad data flag
      mr_bad_data(mr) = pyvar_missing_flag(uvar)

* result is ready to return to the stack
      is_mr(isp) = mr

* got the requested component
      RETURN 2
! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000 RETURN
      END       
