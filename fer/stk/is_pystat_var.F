*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 

*
*  Copy a pystat variable in ferret memory
*
      SUBROUTINE IS_PYSTAT_VAR(memory, *, *, sts)

      IMPLICIT NONE
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'interp_stack.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xpyvar_info.cmn'
      INCLUDE 'xtext_info.cmn'

*     Calling argument declarations
      INTEGER sts
      REAL    memory(mem_blk_size, max_mem_blks)

*     Function declarations
      CHARACTER*128 VAR_CODE

*     Internal variable declarations
      INTEGER cx, cx_var, pyvar, idim, mr_var, mr, sts2
      LOGICAL newvar_needed, need_data

*     Initialize
      cx = is_cx(isp)
      pyvar = cx_variable(cx)

*     Create a context for reading exactly all the data in this variable
      CALL STACK_PTR_UP(cx_stack_ptr, max_context, sts)
      IF ( sts .NE. ferr_ok ) GOTO 5000
      cx_var = cx_stack_ptr
      CALL TRANSFER_CONTEXT(cx, cx_var)
      cx_variable(cx_var) = pyvar
      cx_grid(cx_var) = pyvar_grid_number(pyvar)
      cx_bad_data(cx_var) = pyvar_missing_flag(pyvar)
      DO 100 idim = 1,4
         cx_lo_ss(cx_var, idim) = pyvar_grid_start(idim, pyvar)
         cx_hi_ss(cx_var, idim) = pyvar_grid_end(idim, pyvar)
         cx_by_ss(idim, cx_var) = .TRUE.
         CALL CONFINE_AXIS(idim, cx_var, sts)
         IF ( sts .NE. ferr_ok ) GOTO 4900
         CALL FLESH_OUT_AXIS(idim, cx_var, sts)
         IF ( sts .NE. ferr_ok ) GOTO 4900
  100 CONTINUE

*     Check the axis limits on the data requests.
*     If the limits on the request data exceed the limits of the given
*     data, do NOT issue errors or warnings, or trim the cx context.
*     (So that getting it from cached data will have the same behavior.)
*     Instead, create a variable filled with missing values matching
*     the requested data and copy into it the given data where appropriate.
      newvar_needed = .FALSE.
      need_data = .TRUE.
      DO 200 idim = 1,4
         IF ( cx_lo_ss(cx,idim) .LT. cx_lo_ss(cx_var,idim) ) THEN
*           Limits exceeded so new expanded variable needed
            newvar_needed = .TRUE.
            IF ( cx_hi_ss(cx,idim) .LT. cx_lo_ss(cx_var,idim) ) THEN
*              Completely out of range so do not get or copy the given data
               need_data = .FALSE.
            ENDIF
         ENDIF
         IF ( cx_hi_ss(cx,idim) .GT. cx_hi_ss(cx_var,idim) ) THEN
*           Limits exceeded so new expanded variable needed
            newvar_needed = .TRUE.
            IF ( cx_lo_ss(cx,idim) .GT. cx_hi_ss(cx_var,idim) ) THEN
*              Completely out of range so do not get or copy the given data
               need_data = .FALSE.
            ENDIF
         ENDIF
  200 CONTINUE

*     If needed, create a variable with all the given data
      IF ( need_data ) THEN
*        Reserve and catalog space for exactly all the given data
         CALL CREATE_MEM_VAR(cx_var, mr_var, sts)
         IF ( sts .NE. ferr_ok ) GOTO 4900
*        diagnostic output
         IF (mode_diagnostic) CALL DIAGNOSTIC_OUT(
     .                        'copy pyvar data', mr_var, point_to_mr)
*        Set the bad data flag
         mr_bad_data(mr_var) = pyvar_missing_flag(pyvar)
*        Get all the given data from the python data array object
         CALL COPY_PYSTAT_DATA(memory(1, mr_blk1(mr_var)),
     .                         pyvar_ndarray_obj(pyvar))
      ENDIF


*     If the limits on the requested data exceeded the limits
*     on the actual data, create a new expanded variable; otherwise
*     just use the variable created above.
      IF ( newvar_needed ) THEN
*        Reserve and catalog space for exactly the requested data
         CALL CREATE_MEM_VAR(cx, mr, sts)
         IF ( sts .NE. ferr_ok ) GOTO 4800
*        diagnostic output
         IF (mode_diagnostic) CALL DIAGNOSTIC_OUT(
     .                        'expand pyvar data', mr, point_to_mr)
*        Set the bad data flag
         mr_bad_data(mr) = pyvar_missing_flag(pyvar)
*        Fill completely with the missing value
         CALL FILL_MEMORY(memory, mr_blk1(mr), mr_nblks(mr), 
     .                    mr_bad_data(mr))
         IF ( need_data ) THEN
*           Trim the cx_lo_ss and cx_hi_ss limits in cx_var (given data)
*           so they do not exceed the limits in cx (used to create mr)
            DO 300 idim = 1,4
               IF ( cx_lo_ss(cx_var,idim) .LT. cx_lo_ss(cx,idim) ) THEN
                  cx_lo_ss(cx_var,idim) = cx_lo_ss(cx,idim)
               ENDIF
               IF ( cx_hi_ss(cx_var,idim) .GT. cx_hi_ss(cx,idim) ) THEN
                  cx_hi_ss(cx_var,idim) = cx_hi_ss(cx,idim)
               ENDIF
  300       CONTINUE
*           Copy over the given data where appropriate
            CALL COPY_INTO(memory(1, mr_blk1(mr_var)), mr_var, cx_var,
     .                     memory(1, mr_blk1(mr)),     mr)
*           Delete the variable with the all the given data (cheap to recreate)
            CALL DELETE_VARIABLE(mr_var)
         ENDIF
      ELSE
*        The original data variable is acceptable 
*        (if newvar_needed is .FALSE., need_data must be .TRUE.)
         mr = mr_var
      ENDIF

*     cx_var no longer needed
      CALL STACK_PTR_DN(cx_stack_ptr, cx_stack_ptr_base, sts)
      IF ( sts .NE. ferr_ok ) THEN
         CALL DELETE_VARIABLE(mr)
         GOTO 5000
      ENDIF

*     Result is ready to return to the stack
      is_mr(isp) = mr

*     Got the requested data
      RETURN 2

*     RETURN 1, which is to request a sub-component, is never used ...

*     Error exit
 4800 IF ( need_data ) CALL DELETE_VARIABLE(mr_var)
 4900 CALL STACK_PTR_DN(cx_stack_ptr, cx_stack_ptr_base, sts2)
 5000 RETURN
      END       
