c*
c* ef_get_coordinates_sub.F
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

      SUBROUTINE EF_GET_COORDINATES_SUB(cx_list, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER cx_list(EF_MAX_ARGS), iarg
      INTEGER ss_lo_lim, ss_hi_lim
      INTEGER idim, grid, i

      REAL    values( ss_lo_lim:ss_hi_lim )
      REAL*8  TM_WORLD


*     defining grid for output data
      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values( i ) = TM_WORLD( i,grid,idim,box_middle )
 100  continue

      RETURN 
      END
