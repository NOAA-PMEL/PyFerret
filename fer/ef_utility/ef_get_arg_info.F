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

      SUBROUTINE ef_get_arg_info(id, iarg, name, title, units )

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, iarg
      INTEGER cat, var

      CHARACTER*(*) name, title, units
      CHARACTER var_code*24, var_title*128, var_units*32

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL ef_get_cx_list(cx_list)

      cat = cx_category(cx_list(iarg))
      var = cx_variable(cx_list(iarg)) 

      name  = var_code(cat,var)
      title = var_title(cx_list(iarg))
      units = var_units(cx_list(iarg))


      RETURN 
      END
