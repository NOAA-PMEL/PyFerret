c*
c* ef_get_arg_info.F
c*
c* Jonathan Callahan
c* Sep 28th 1998
c*
c*

c*
c* This returns the name, title and dataset associated 
c* with an argument.
c*

      SUBROUTINE EF_GET_ARG_INFO(id, iarg, name, title, units )

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, iarg
      INTEGER cat, var

      CHARACTER name*24, title*128, units*32
      CHARACTER VAR_CODE*24, VAR_TITLE*128, VAR_UNITS*32

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      cat = CX_CATEGORY(cx_list(iarg))
      var = CX_VARIABLE(cx_list(iarg)) 

      name  = VAR_CODE(cat,var)
      title = VAR_TITLE(cx_list(iarg))
      units = VAR_UNITS(cx_list(iarg))


      RETURN 
      END
