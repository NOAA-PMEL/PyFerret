c*
c* ef_get_arg_ss_extremes_sub.F
c*
c* Jonathan Callahan
c* Mar 27th 1998
c*
c*

c*
c* This subroutine is called by code inside EF_ExternalUtil.c. It will
c* return the minimum and maximum subscripts for a grid.  
c*

      SUBROUTINE EF_GET_ARG_SS_EXTREMES_SUB(cx_list, num_args,
     .     ss_min, ss_max)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER cx_list(EF_MAX_ARGS), num_args
      INTEGER ss_min(4,EF_MAX_ARGS), ss_max(4,EF_MAX_ARGS)

      INTEGER grid, idim, iarg

      DO 200 iarg=1,num_args

         grid = cx_grid( cx_list(iarg) )

         DO 100 idim=1,4

	   CALL GRID_SUBSCRIPT_EXTREMES(ss_min(idim,iarg), 
     .           ss_max(idim,iarg), grid, idim)

 100     CONTINUE
 200  CONTINUE

      RETURN 
      END
