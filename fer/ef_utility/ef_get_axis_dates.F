c*
c* ef_get_axis_dates.F
c*
c* Ansley Manke
c* Dec 9 1998
c*
c*

c*
c* This returns the date buffer associated with the 
c* time axis of an argument.
c*

      SUBROUTINE EF_GET_AXIS_DATES(id, iarg, tax, numtimes, datebuf )

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xtm_grid.cmn_text'

      INTEGER id, iarg, l
      INTEGER grid
      INTEGER numtimes, seconds

      PARAMETER (seconds=6)

      CHARACTER*(*) datebuf(*)
      REAL*8  tax(*), tstep_r8

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL EF_GET_CX_LIST(cx_list)
      grid = CX_GRID(cx_list(iarg))

      DO 100 l = 1, numtimes
         tstep_r8 = tax(l)
         CALL tstep_to_date ( grid, tstep_r8, seconds, datebuf(l) )
100   CONTINUE


      RETURN 
      END
