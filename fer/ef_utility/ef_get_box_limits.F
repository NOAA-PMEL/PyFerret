c*
c* ef_get_box_limits.F
c*
c* Ansley Manke
c* Jan 12th 1999
c*Based on ef_get_coordinates.F
c*

c*
c* This fills in the box_limits of an axis in a grid.  
c*
c* JC_TODO: What about REAL*8 values being passed back to C
c*

      SUBROUTINE EF_GET_BOX_LIMITS(id, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values_lo, values_hi )

      INCLUDE 'ferret.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, iarg, idim
      INTEGER ss_lo_lim, ss_hi_lim
      INTEGER grid, i

      REAL*8  values_lo( ss_lo_lim:ss_hi_lim )
      REAL*8  values_hi( ss_lo_lim:ss_hi_lim )
      REAL*8  TM_WORLD

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL EF_GET_CX_LIST(cx_list)

*     defining grid for output data
      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values_lo( i ) = TM_WORLD( i,grid,idim,box_lo_lim )
 100  continue

      DO 200 i = ss_lo_lim, ss_hi_lim
         values_hi( i ) = TM_WORLD( i,grid,idim,box_hi_lim )
 200  continue

      RETURN 
      END
