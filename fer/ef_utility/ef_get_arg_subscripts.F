c*
c* ef_get_arg_subscripts.F
c*
c* Jonathan Callahan
c* Feb 19th 1998
c*
c*

c*
c* Initialize the context limits the EF will use.  
c*

      SUBROUTINE EF_GET_ARG_SUBSCRIPTS(id, arg_lo_ss, arg_hi_ss, 
     .     arg_incr)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg_lo_ss(4,EF_MAX_ARGS+1), 
     .     arg_hi_ss(4,EF_MAX_ARGS+1), arg_incr(4,EF_MAX_ARGS+1)

      INTEGER idim, iarg, num_reqd_args
      INTEGER cx_list(EF_MAX_ARGS)
      INTEGER efcn_get_num_reqd_args

      num_reqd_args = efcn_get_num_reqd_args(id)


c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      DO 200 idim=1,4

         DO 100 iarg=1, num_reqd_args

            arg_lo_ss(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
            arg_hi_ss(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
            IF (arg_lo_ss(idim,iarg) .EQ. arg_hi_ss(idim,iarg)) 
     .           THEN
               arg_incr(idim, iarg) = 0
            ELSE
               arg_incr(idim, iarg) = 1
            ENDIF

 100     CONTINUE

 200  CONTINUE

      RETURN 
      END
