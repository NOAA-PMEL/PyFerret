	REAL FUNCTION BOX_SIZE( isubscript, grid, idim )

* given a subscript value on a grid and an axis return the length of the
* grid box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 12/9/86
* revision 1.0 - 3/10/87 - line data stored in COMMON/~VARIABLES/
* revision 0.2 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 0.3 - 2/18/88 - incorporated "regular" lines
* revision 1.0 - 1/12/88 - new TM IO library stores box bounds, not sizes
* Unix/RISC port - 1/91 - cant use "/star" in documentation
*                  9/91 - incorporated (missing) modulo axis logic
* V4.20 - 10/95 - remove references to line_mem to permit recursive line defs
*	   1/96 - bug fix: only look at line_parent for dynamic lines
*		           (code would be sweeter if ALL lines had line_parent)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	isubscript, grid, idim

* internal variable declarations:
	LOGICAL recursive
	INTEGER	iaxis, isub0, isub
	REAL*8	TM_WORLD, midpoint, lo_point, hi_point

* initialize
	iaxis = grid_line( idim, grid )
	IF ( iaxis .EQ. mnormal .OR. iaxis .EQ. munknown ) THEN
	   BOX_SIZE = bad_val4
	   RETURN
	ENDIF

	IF ( line_regular( iaxis ) ) THEN
* regularly spaced points all have same grid box size
	   BOX_SIZE = SNGL( line_delta( iaxis ) )

	ELSE
* irregularly spaced points
* keep subscript within bounds ( between 1 and dimension of axis ) by ...
           IF ( line_modulo( iaxis ) ) THEN
* ... modulo-ing to the appropriate value
              isub0 = MOD(isubscript-1,line_dim(iaxis)) + 1
           ELSE
* ... truncating to the appropriate value
	      isub0 = MIN( line_dim(iaxis), MAX( 1, isubscript ) )
           ENDIF

* only dynamic lines have the possibility of recursive definitions
* (since FORTRAN doesn't protect "AND" clauses this test in 2 steps)
	   recursive = iaxis .GT. max_lines
	   IF ( recursive ) recursive = line_parent(iaxis) .NE. 0 

* use TM_WORLD to compute BOX_SIZE for recursive defs (10/95)
	   IF ( recursive ) THEN
* ... every nth point from an irregular axis --> use midpoints as box bounds
	      IF (isub0.EQ.1 .AND. .NOT.line_modulo(iaxis) ) THEN
	         lo_point =  TM_WORLD(isub0,   grid, idim, box_lo_lim)
	      ELSE
	         lo_point = (TM_WORLD(isub0,   grid, idim, box_middle) 
     .			  +  TM_WORLD(isub0-1, grid, idim, box_middle)) / 2
	      ENDIF
	      IF ( isub0.EQ.line_dim(iaxis) 
     .	     .AND. .NOT.line_modulo(iaxis) ) THEN
	         hi_point =  TM_WORLD(isub0,   grid, idim, box_hi_lim)
	      ELSE
	         hi_point = (TM_WORLD(isub0,   grid, idim, box_middle) 
     .			  +  TM_WORLD(isub0+1, grid, idim, box_middle)) / 2
	      ENDIF
	      BOX_SIZE = SNGL(hi_point - lo_point)
	   ELSE
	      BOX_SIZE = SNGL( TM_WORLD(isub0, grid, idim, box_hi_lim)
     .		             - TM_WORLD(isub0, grid, idim, box_lo_lim) )
	   ENDIF

	ENDIF

	RETURN
	END
