c*
c* ef_get_axis_info.F
c*
c* Jonathan Callahan
c* Sep 28th 1998
c*
c*

c*
c* This returns the name, title and units associated with
c* each axis of an argument.
c*

      SUBROUTINE EF_GET_AXIS_INFO(id, iarg, name, units,
     .  backward, modulo, regular )

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'tmap_format/tmap_dims.parm'
      INCLUDE 'tmap_format/xtm_grid.cmn_text'

      INTEGER id, iarg
      INTEGER grid, axis, idim

      CHARACTER name(4)*16, units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)
      LOGICAL BKWD_AXIS

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      grid = CX_GRID(cx_list(iarg))

      DO 100 idim=1, 4
        axis = GRID_LINE(idim, grid) 
        IF (axis .EQ. munknown) THEN
          name(idim) = 'unknown'
          units(idim) = 'none'
        ELSE IF (axis .EQ. mnormal) THEN
          name(idim) = 'normal'
          units(idim) = 'none'
        ELSE
          name(idim)  = line_name(axis)
          units(idim) = line_units(axis)
          backward(idim) = BKWD_AXIS(idim, grid)
          modulo(idim)   = line_modulo(axis)
          regular(idim)  = line_regular(axis)
        ENDIF
100   CONTINUE


      RETURN 
      END
