c*
c* ef_get_coordinates.F
c*
c* Jonathan Callahan
c* Mar 10th 1998
c*
c*

c*
c* This fills in the coordinates of an axis in a grid.  
c*
c* JC_TODO: What about REAL*8 values being passed back to C
c*

      SUBROUTINE EF_GET_COORDINATES(id, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, idim
      INTEGER ss_lo_lim, ss_hi_lim
      INTEGER grid, i

      REAL*8  values( ss_lo_lim:ss_hi_lim )
      REAL*8  TM_WORLD

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

*     defining grid for output data
      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values( i ) = TM_WORLD( i,grid,idim,box_middle )
 100  continue

      RETURN 
      END
