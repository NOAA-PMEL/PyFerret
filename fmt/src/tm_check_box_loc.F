	LOGICAL FUNCTION TM_Check_box_loc (nline)
*
* Checks to make sure a data point is enclosed by its box on an axis
*
* revision 0.00 - 05/26/88 - by M. Verschell for PMEL/TMAP
*
* Argument
	INTEGER*4	nline
*
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
	INTEGER*4	ipt1, npts, ibx1, i
*
	ipt1 = line_subsc1(nline)
	npts = line_dim(nline)
	ibx1 = ipt1 + npts

	DO 100 i=0,npts-1
	  IF (line_mem(ipt1+i) .LT. line_mem(ibx1+i) .OR.
     .	      line_mem(ipt1+i) .GT. line_mem(ibx1+i+1)) THEN
	    TM_Check_box_loc = .FALSE.
	    RETURN
	  ENDIF
  100	CONTINUE

	TM_Check_box_loc = .TRUE.
	RETURN
	END
