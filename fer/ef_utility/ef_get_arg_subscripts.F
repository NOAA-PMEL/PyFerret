c*
c* ef_get_arg_subscripts.F
c*
c* Jonathan Callahan
c* Feb 19th 1998
c*
c* V510 12/99 *sh* - cope with unspecified limits (default=full axis range)
c*               and limits embedded inside of user variable definitions
c*               as in LET a = sst[L=5:9]; LIST FFT(a) (info from uvar_*)


c*
c* Initialize the context limits the EF will use.  
c*

      SUBROUTINE EF_GET_ARG_SUBSCRIPTS(id, arg_lo_ss, arg_hi_ss, 
     .     arg_incr)

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

* argument declarations
      INTEGER id, arg_lo_ss(4,EF_MAX_ARGS+1), 
     .     arg_hi_ss(4,EF_MAX_ARGS+1), arg_incr(4,EF_MAX_ARGS+1)


* internal variable declarations
      LOGICAL embedded_limits
      INTEGER idim, iarg, num_reqd_args, cat, var
      INTEGER cx_list(EF_MAX_ARGS), ax_src(4)
      INTEGER efcn_get_num_reqd_args

      num_reqd_args = efcn_get_num_reqd_args(id)
      CALL efcn_get_axis_will_be(id, ax_src)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      DO 200 idim=1,4

         DO 100 iarg=1, num_reqd_args

            IF ( cx_given(idim,cx_list(iarg)) ) THEN
* limits are explicit in the top-level expression
              arg_lo_ss(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
              arg_hi_ss(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
            ELSE
* are there explicit limits embedded within user-var definitions?
              cat = cx_category( cx_list(iarg) )
              var = cx_variable( cx_list(iarg) )
              IF ( cat .EQ. cat_user_var ) THEN
                 embedded_limits =
     .                 uvar_given(idim,var) .GE. uvlim_gvn_gnl
     .           .AND. uvar_given(idim,var) .LE. uvlim_gvn_xact
              ELSE
                 embedded_limits = .FALSE.
              ENDIF
              IF ( embedded_limits ) THEN
* ... get embedded explicit limits from the user-variable definition
                arg_lo_ss(idim,iarg) = uvar_lo_ss(var, idim)
                arg_hi_ss(idim,iarg) = uvar_hi_ss(var, idim)
              ELSE
                IF (ax_src(idim) .EQ. IMPLIED_BY_ARGS) THEN
* ... result axis inherits limits from SET REGION and command qualifiers, etc.
* NOTE: ARG LIMITS WILL BE WRONG IF USED DURING GRID-CREATION
*       (which should never happen ...)
                  arg_lo_ss(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
                  arg_hi_ss(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
                ELSE
* ... result axis is imposed by the EF and this arg has no explicit limits
*     so infer the limits from the nature of the variable
                  IF (  cat .EQ. cat_file_var ) THEN
*    ... get limits of file variable (possibly less than full-axis)
                    CALL VAR_SS_LIMS( idim, cx_list(iarg),
     .                                arg_lo_ss(idim,iarg),
     .                                arg_hi_ss(idim,iarg) )
                  ELSE
*    ... assume the full span of the defining axis as the default range
                    CALL GRID_SUBSCRIPT_EXTREMES_NO_MOD(
     .                  arg_lo_ss(idim,iarg),
     .                  arg_hi_ss(idim,iarg),
     .                  cx_grid(cx_list(iarg)), idim)
                  ENDIF
                ENDIF
              ENDIF
            ENDIF

* fill in the delta to indicate if a range is available
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
