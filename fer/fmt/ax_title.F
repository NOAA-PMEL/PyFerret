 	CHARACTER*(*) FUNCTION AX_TITLE( idim, grid, len )

* produce a title string with units suitable for an axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 8/25/88
* V320: 12/19/94 - recognize millibars and decibars as special axes
*		(kludge: use "m" for meters instead of exact unit string.
*			This because of "METERS" in the old grid files around.)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xunits.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, grid, len
	LOGICAL	GEOG_LABEL

* internal variable declarations:
	INTEGER	TM_LENSTR1, i, line, ulen
	CHARACTER*16 unit_name

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

	line = grid_line( idim, grid )

	IF ( GEOG_LABEL( idim, grid ) ) THEN
	   AX_TITLE = axis_title( idim )
	   len = axis_title_len( idim )
! special options for vertical axes - 12/19/94
	   IF ( idim .EQ. z_dim ) THEN
	      IF ( line_unit_code(line) .EQ. pun_millibars 
     .        .OR. line_direction( line ) .EQ. 'DU'     ) THEN
	         AX_TITLE = "HEIGHT"
	         len = 6
	      ENDIF
	      IF ( line_unit_code(line) .EQ. pun_meters ) THEN
	         unit_name = 'm'		! kludge for "METERS"
	      ELSE
	         unit_name = line_units(line)
	      ENDIF
	      ulen = TM_LENSTR1( unit_name )
	      AX_TITLE = AX_TITLE(1:len)//' ('//unit_name(:ulen)//')'
	      len = len + 3 + ulen
	   ENDIF	      

	ELSE
	   AX_TITLE = ww_dim_name( idim )
	   len = 1
	   IF ( line .NE. mnormal .AND. line .NE. munknown ) THEN
	      IF ( line_units(line) .NE. ' ' ) THEN
	         ulen = TM_LENSTR1( line_units(line) )
	         AX_TITLE = AX_TITLE(1:1)//' ('//line_units(line)(:ulen)//')'
	         len = 4 + ulen
	      ENDIF
	   ENDIF
	ENDIF

	RETURN
	END
