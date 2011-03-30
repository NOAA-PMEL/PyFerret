* copy a pystat variable in ferret memory

      SUBROUTINE IS_PYSTAT_VAR(memory, *, *, sts)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'interp_stack.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xpyvar_info.cmn'
      INCLUDE 'xtext_info.cmn'

* calling argument declarations:
      INTEGER sts
      REAL    memory(mem_blk_size, max_mem_blks)

* function declarations:
      CHARACTER*128 VAR_CODE

* internal variable declarations:
      INTEGER cx, cx_var, pyvar, idim, mr, sts2
      LOGICAL give_warning

* initialize
      cx = is_cx(isp)
      pyvar = cx_variable(cx)

* create a context describing this variable
      CALL STACK_PTR_UP(cx_stack_ptr, max_context, sts)
      cx_var = cx_stack_ptr
      IF ( sts .NE. ferr_ok ) GOTO 5100
      CALL TRANSFER_CONTEXT(cx, cx_var)
      cx_variable(cx_var) = pyvar
      cx_grid(cx_var) = pyvar_grid_number(pyvar)
      cx_bad_data(cx_var) = pyvar_missing_flag(pyvar)
      DO 100 idim = 1,4
          cx_lo_ss(cx_var, idim) = pyvar_grid_start(idim, pyvar)
          cx_hi_ss(cx_var, idim) = pyvar_grid_end(idim, pyvar)
          cx_by_ss(idim, cx_var) = .TRUE.
          CALL CONFINE_AXIS(idim, cx_var, sts)
          IF ( sts .NE. ferr_ok ) GOTO 5000
          CALL FLESH_OUT_AXIS(idim, cx_var, sts)
          IF ( sts .NE. ferr_ok ) GOTO 5000
  100 CONTINUE

* reserve and catalog space for the result
      CALL CREATE_MEM_VAR(cx_var, mr, sts)
      IF ( sts .NE. ferr_ok ) GOTO 5000

* diagnostic output
      IF (mode_diagnostic) 
     .    CALL DIAGNOSTIC_OUT('pystat_var', mr, point_to_mr)

      CALL COPY_PYSTAT_DATA(memory(1,mr_blk1(mr)),
     .                      pyvar_ndarray_obj(pyvar))

* set the bad data flag
      mr_bad_data(mr) = pyvar_missing_flag(pyvar)

* result is ready to return to the stack
      is_mr(isp) = mr

      DO 200 idim = 1,4
          give_warning = .FALSE.
          IF ( cx_lo_ss(cx,idim) .LT. cx_lo_ss(cx_var,idim) ) THEN
              IF ( cx_hi_ss(cx,idim) .LT. cx_lo_ss(cx_var,idim) ) THEN
*                 Error - the requested data is completely out of range
                  GOTO 5200
              ENDIF
              IF ( (cx_lo_ss(cx_last,idim) .NE. unspecified_int4)
     .             .OR.  cx_given(idim,cx) ) THEN
*                 Warn - explicitly requested more data than what exists
                  give_warning = .TRUE.
              ENDIF
              cx_lo_ss(cx,idim) = cx_lo_ss(cx_var,idim)
              cx_lo_ww(idim,cx) = cx_lo_ww(idim,cx_var)
          ENDIF
          IF ( cx_hi_ss(cx,idim) .GT. cx_hi_ss(cx_var,idim) ) THEN
              IF ( cx_lo_ss(cx,idim) .GT. cx_hi_ss(cx_var,idim) ) THEN
*                 Error - the requested data is completely out of range
                  GOTO 5200
              ENDIF
              IF ( (cx_hi_ss(cx_last,idim) .NE. unspecified_int4)
     .             .OR.  cx_given(idim,cx) ) THEN
*                 Warn - explicitly requested more data than what exists
                  give_warning = .TRUE.
              ENDIF
              cx_hi_ss(cx,idim) = cx_hi_ss(cx_var,idim)
              cx_hi_ww(idim,cx) = cx_hi_ww(idim,cx_var)
          ENDIF
          IF ( give_warning ) THEN
              CALL WARN('Given range exceeds actual for the '
     .                  // ww_dim_name(idim) // ' axis of ' // 
     .                  VAR_CODE(cat_pystat_var,pyvar))
          ENDIF
 200  CONTINUE

      CALL STACK_PTR_DN(cx_stack_ptr, cx_stack_ptr_base, sts)
      IF ( sts .NE. ferr_ok ) GOTO 5100

* got the requested data
      RETURN 2
* RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000 CALL STACK_PTR_DN(cx_stack_ptr, cx_stack_ptr_base, sts2)
 5100 RETURN
 5200 CALL ERRMSG(ferr_limits, sts, ww_dim_name(idim) //
     .            ' axis of ' // VAR_CODE(cat_pystat_var,pyvar), *5000)
      END       
