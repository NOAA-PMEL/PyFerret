      CHARACTER*(*) FUNCTION TM_GET_CAL (grid)

C     Return the calendar name associated with a grid
C     Ansley Manke 3/01

#ifdef unix
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'ferret.parm'
      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'
#endif

C  Calling argument declarations:
      INTEGER grid

C  Internal argument declarations:
      INTEGER tax

      tax = grid_line (t_dim, grid)

      TM_GET_CAL = line_cal_name (tax)

      RETURN
      END
