	SUBROUTINE SHOW_DSG_RANGES ( dset, showing_grid, lun )
	
* For DSGF dataset, SHOW DATA/FULL or SHOW GRID, list the
* coordinate ranges lon, lat, z, time for the observations/

* programmer - Ansley
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  3/3/2020
*
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'xtext_info.cmn'

	INTEGER		dset, lun
	LOGICAL		showing_grid

	INTEGER		STR_DNCASE,TM_LENSTR1, 
     .			igrid, ndec, idim, dlen, status
	REAL		TM_WORLD, lo, hi
	CHARACTER	buff*16, line*255
	
	CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)
	IF (showing_grid) THEN
	   line = 'Dataset coordinates:'
	   CALL SPLIT_LIST(pttmode_explct, lun, line, 20)
	ENDIF

	igrid =  dsg_xlate_grid(dset)
	ndec = 5
	
	DO idim = 1, t_dim
		
	   lo = TM_WORLD( 1, igrid, idim, box_middle )
	   hi = TM_WORLD( line_dim(grid_line(idim, igrid)), igrid, idim, box_middle )
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
