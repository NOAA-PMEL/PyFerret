	DOUBLE PRECISION FUNCTION TWORLD( subsc, grid, idim, where_in_box )

* convert a subscript value on a grid into a "world" coordinate
* if the axis is a valid time axis then convert the time step value into
* seconds from jan-1-0000:00:00 so that all time axes are on an absolute scale

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/26/88
* revision 0.1 -12/30/88 - added where_in_box
* V230:  7/20/92 - check for valid time axis T0

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	subsc, grid, idim, where_in_box

* internal variable declarations:
        LOGICAL TM_DATE_OK
        INTEGER line
	REAL*8	TM_WORLD, TSTEP_TO_SECS, ww

* standard conversion into world coordinates
	ww = TM_WORLD( subsc, grid, idim, where_in_box )
	TWORLD = ww

* special treatment of calendar time axis
	IF ( idim .EQ. t_dim ) THEN
           line = grid_line( idim, grid )
           IF ( TM_DATE_OK( line_t0(line) )  )
     .          TWORLD = TSTEP_TO_SECS( grid, ww )
	ENDIF
	RETURN
	END
