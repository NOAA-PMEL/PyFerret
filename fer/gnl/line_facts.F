	SUBROUTINE LINE_FACTS( lun, idim, grid, orient )

* print basic facts about an axis defined in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88 - extracted from SHOW_GRID - axis limits --> box_middle
* revision 0.1 - 8/16/88 - elimiinated units, expanded title field
* revision 0.2 - 2/6/89 - added modulo axis indicator 
* V200:  9/22/89 - added "orient" for axes without particular orientations
* SUN port 3/92 *sh* - "T" format specifiers cannot move to left
*                    - changed format 3020
* V230:  6/22/92 - allow for long axis names by truncating the title field
*        1/93 - corrected left-over bug in format 3020 from SUN port A15->A14
* V320: 12/20/94 - Put "(-)" along with "m" and "r" so it doesn't get
*		truncated now that DEPTH(units) is the label
*       12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'          ! 12/94 SPLIT_LIST BUFFER
#endif

* calling argument declarations
	INTEGER	  lun, idim, grid, orient

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL, calendar
	INTEGER TM_LENSTR1, lo, hi, i, line, dlen, ndec, status, nlen
	CHARACTER AX_TITLE*32, 	lo_str*20, hi_str*20, direction*1,
     .		  title*32, reg*1, modflag*1, truncated*1
	REAL*8	TM_WORLD, lo_val8, hi_val8

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )

* pull line from grid table
	line = grid_line( idim, grid )

* line ok ?
	risc_buff = ' '
	IF ( line .EQ. munknown ) THEN
	   title = 'unknown'
	   WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSEIF ( line .EQ. mnormal ) THEN
	   title = 'normal'
	   WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSE

* get nicely formatted limits (end box outer edges)
	   calendar = idim .EQ. t_dim .AND. GEOG_LABEL( idim, grid )	
	   lo_val8 = TM_WORLD( 1, grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 		! accurate to minutes
	   ELSE
	      ndec = -sig_figs	! 5 sig places w/out extra zeros
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )
	   hi_val8 = TM_WORLD( line_dim(line), grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 
	   ELSE
	      ndec = -sig_figs
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )

* axis oriented backwards ?
	   IF ( BKWD_AXIS(idim, grid) ) THEN
	      direction = '-'
	   ELSE
	      direction = ' '
	   ENDIF

* axis title
	   IF ( orient .EQ. 0 ) THEN
	      title = ' '
	      dlen = 1
	   ELSE
	      title = AX_TITLE( idim, grid, dlen )
	   ENDIF
!	   title = title(:dlen)//direction	! 12/94 removed

* regularly spaced ?
	IF ( line_regular( line ) ) THEN
	  reg = 'r'
	ELSE
	  reg = 'i'
	ENDIF

* modulo ?
	IF ( line_modulo( line ) ) THEN
	  modflag = 'm'
	ELSE
	  modflag = ' '
	ENDIF

* allow for long axis names (6/92)
        nlen = MAX( TM_LENSTR1(line_name(line)), 9 )

* write info
	   WRITE ( risc_buff, 3020 )	line_name ( line )(:nlen),
     .				title (:41-nlen),
     .				line_dim  ( line ),
     .				modflag,
     .				reg,
     .				direction,
     .				lo_str,
     .				hi_str
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF
	RETURN

 3010	FORMAT ( T2,A8,T12,A1 )
 3020	FORMAT ( T2,A,1X,A14,T27,I7,3A1,T39,A20,T60,A20 )
! old V220: 3020 FORMAT ( T2,A8,T12,A15,T27,I7,2A1,T39,A20,T60,A20 )

	END
