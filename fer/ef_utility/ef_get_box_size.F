c*
c* ef_get_box_size.F
c*
c* Jonathan Callahan
c* Mar 10th 1998
c*
c*

c*
c* This fills in the coordinates of an axis in a grid.  
c*

      SUBROUTINE EF_GET_BOX_SIZE(id, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, idim, ss_lo_lim, ss_hi_lim
      INTEGER grid, i

      REAL    values( ss_lo_lim:ss_hi_lim )
      REAL    BOX_SIZE

      INTEGER cx_list(EF_MAX_ARGS)
 
c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values( i ) = BOX_SIZE( i,grid,idim )
 100  continue

      RETURN 
      END
