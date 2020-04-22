	SUBROUTINE SHOW_DSG_RANGES ( dset, showing_grid, lun )
	
* For DSGF dataset, SHOW DATA/FULL or SHOW GRID, list the
* coordinate ranges lon, lat, z, time for the observations/

* programmer - Ansley
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  3/3/2020
* 4/20/2020  *acm* Use MASKED_DSG_RANGE to get the range in each direction
*            including nice longitudes corrected for modulo lon
*
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'xtext_info.cmn'

	INTEGER		dset, grid, lun
	LOGICAL		showing_grid

	INTEGER		STR_DNCASE,TM_LENSTR1, 
     .			igrid, ndec, cx, idim, dlen, status
	REAL		TM_WORLD, lo, hi
	CHARACTER	buff*16, line*255
	
	CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)
	IF (showing_grid) THEN
	   line = 'Dataset coordinates:'
	   CALL SPLIT_LIST(pttmode_explct, lun, line, 20)
	ENDIF

	ndec = 5
	igrid =  dsg_xlate_grid(dset)
	cx = cx_none

	DO idim = 1, t_dim

	   lo = unspecified_val8
	   hi = unspecified_val8
	   CALL MASKED_DSG_RANGE (dset, cx, idim, lo, hi)
	   IF (lo .EQ. unspecified_val8) CYCLE
	
	   CALL TRANSLATE_TO_WORLD( lo, idim, igrid, ndec, line )
	   dlen = TM_LENSTR1(line)
	   line(dlen+1:) = ' to ' 
	   
	   CALL TRANSLATE_TO_WORLD( hi, idim, igrid, ndec, line(dlen+5:) )
	   dlen = TM_LENSTR1(line)

	   status = STR_DNCASE(buff, axis_title(idim))

	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			'  '//buff(:TM_LENSTR1(buff))//' range: '//line(:dlen), 0) 

	ENDDO
	
	CALL SPLIT_LIST(pttmode_explct, lun,'  ', 0) 
	

	RETURN
	END
