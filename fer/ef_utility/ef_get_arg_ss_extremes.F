c*
c* ef_get_arg_ss_extremes.F
c*
c* Jonathan Callahan
c* Mar 27th 1998
c*

c*
c* Return the minimum and maximum subscripts for a grid.  
c*

      SUBROUTINE EF_GET_ARG_SS_EXTREMES(id, num_args, ss_min, ss_max)

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, num_args, ss_min(4,EF_MAX_ARGS), 
     .     ss_max(4,EF_MAX_ARGS)

      INTEGER grid, idim, iarg
      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      DO 200 iarg=1,num_args

         grid = cx_grid( cx_list(iarg) )

         DO 100 idim=1,4

	   CALL GRID_SUBSCRIPT_EXTREMES(ss_min(idim,iarg), 
     .           ss_max(idim,iarg), grid, idim)

 100     CONTINUE
 200  CONTINUE

      RETURN 
      END
