	SUBROUTINE TM_COPY_LINE ( source, dest )
*
* Copies line information from source to dest
*
* revision 0.00 - 11/22/88
* 10/95 - check to see if a dynamic line -- some variables don't get copied
*  1/96 - repaired bug (incomplete test) in 10/95 check

* Arguments
	INTEGER*4	source, dest
*
* Parameters and commons
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
	line_name(dest)      = line_name(source)
	line_dim(dest)       = line_dim(source) 
	line_units(dest)     = line_units(source) 
	line_unit_code(dest) = line_unit_code(source) 
	line_direction(dest) = line_direction(source) 
	line_regular(dest)   = line_regular(source) 
	line_modulo(dest)    = line_modulo(source) 
	line_start(dest)     = line_start(source) 
	line_delta(dest)     = line_delta(source) 
	line_t0(dest)        = line_t0(source) 
	line_tunit(dest)     = line_tunit(source) 

	IF ( dest.LE.max_lines .AND. source.LE.max_lines) THEN
* non-dynamic lines, only
	   line_subsc1(dest)    = line_subsc1(source)
	ELSEIF ( dest.GT.max_lines .AND. source.GT.max_lines) THEN! ELSEIF 1/96
* dynamic lines, only
	   line_class (dest)	= line_class (source)
	   line_parent(dest)	= line_parent(source)
	ENDIF

	RETURN
	END
