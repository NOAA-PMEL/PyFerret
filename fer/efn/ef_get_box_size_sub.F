c*
c* ef_get_box_size_sub.F
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

      SUBROUTINE EF_GET_BOX_SIZE_SUB(cx_list, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER cx_list(EF_MAX_ARGS), iarg
      INTEGER ss_lo_lim, ss_hi_lim
      INTEGER idim, iarg, grid, i

      REAL    values( ss_lo_lim:ss_hi_lim )
      REAL    BOX_SIZE


*     defining grid for output data
      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values( i ) = BOX_SIZE( i,grid,idim )
 100  continue

      RETURN 
      END
