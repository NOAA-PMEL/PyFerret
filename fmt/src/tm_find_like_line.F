	INTEGER*4 FUNCTION TM_Find_like_line ( iline )
*
* Check to see if an existing line is identical to the line pointed to by
* "iline".
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 05/18/88
* revision 0.01 - 11/21/88 - variable used check changed to specific char_initXX
* revision 0.02 - 09/21/89 - *sh* irregular line coordinate comparison fixed
* revision 0.02 - 2/92 - *sh* removed guts of routine to TM_SAME_LINE_DEF

* Include files
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

* Argument definitions
	INTEGER*4	iline

* Local definitions
        LOGICAL         TM_SAME_LINE_DEF
	INTEGER*4	i

* Loop through defined lines
	DO 100 i = 1, max_lines

	  IF (i.NE.iline .AND. line_name(i).NE.char_init16) THEN

             IF (TM_SAME_LINE_DEF( i, iline ) ) THEN
	        TM_Find_like_line = i
                RETURN
	     ENDIF

          ENDIF

  100	CONTINUE

* No identical line exists
	TM_Find_like_line = unspecified_int4

 	RETURN
	END
