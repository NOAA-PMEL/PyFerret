	SUBROUTINE GCF_IMPOSE_AXES( cx_base, mrg_cx, uvar, status )

* at this point a grid has been determined which lacks only those
* axes which are to be imposed by a grid changing variable. This routine
* completes that grid and the uvar_given associated with it.

* V450: 1/97

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
	INTEGER		cx_base, mrg_cx, uvar, status

* local variable declarations
	INTEGER WHICH_GCFCN
	INTEGER	gcfcn, idim, imp_ax, mrg_ax, mrg_grd,
     .		axis_source(4), imposed_axes(4)


* initialize
* which grid changing variable?
	gcfcn = WHICH_GCFCN(uvar)

* which axes are involved?
	CALL GCF_GET_AXIS_SRCS( gcfcn, 1, axis_source )

* get the replacement axes
	CALL GCF_GET_IMPOSED_AXES( gcfcn, cx_base, imposed_axes, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* get the current grid
	mrg_grd = cx_grid( mrg_cx )


* loop through the axes
	DO 500 idim = 1,4

	   mrg_ax  = grid_line ( idim, mrg_grd )

	   IF (    axis_source(idim).EQ.pgc_ignore_axis) THEN
	      GOTO 500
	   ELSEIF (axis_source(idim).EQ.pgc_impose_axis) THEN
	      imp_ax = imposed_axes(idim)
	      IF ( imp_ax .GT. max_lines ) CALL TM_USE_LINE(imp_ax)
	      IF ( mrg_ax .GT. max_lines ) CALL TM_DEALLO_DYN_LINE(mrg_ax)
	      grid_line( idim, mrg_grd ) = imp_ax
	      IF ( imp_ax .EQ. mnormal ) THEN
	         uvar_given(idim,uvar) = uvlim_irrelevant
	      ELSE
* ... imposed axis has no limit info in the definition
	         uvar_given(idim,uvar) = uvlim_needed
	      ENDIF
	   ELSE
* ... it is evidently, a mergible axis ... continue on
	      CONTINUE
	   ENDIF	      

 500	CONTINUE

* successfully merged
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
