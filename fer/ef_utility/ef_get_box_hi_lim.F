c*
c* ef_get_box_hi_lim.F
c*
c* Ansley Manke
c* Jan 12th 1999
c*Based on ef_get_coordinates.F
c*

c*
c* This fills in the box_hi_lim of an axis in a grid.  
c*
c* JC_TODO: What about REAL*8 values being passed back to C
c*

      SUBROUTINE EF_GET_BOX_HI_LIM(id, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.cmn'

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
         values( i ) = TM_WORLD( i,grid,idim,box_hi_lim )
 100  continue

      RETURN 
      END
