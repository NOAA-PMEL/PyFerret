	SUBROUTINE SHOW_DATA_SET_VARS( lun, dset, full )

* describe the variables in the named data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/10/89 - based on SHOW_DATA_SET 0.2
*			  - with added world coordinate output
* V200:  8/31/90 - changed field widths for subscript limits
* Unix/RISC port 3/11/91 *sh* - eliminated "unaligned data access" errors when
*                               writing variables
* V230: ?? - formatting changes to accomodate indices of more than 4 digits
*     6/92 - locate a grid using this time axis for "time range"
*     7/92 - bypass listing of variables named "-" (EZ data sets)
*    11/92 - increased length of variable names 8 --> 24 characters
* V301: 3/94 - bug fix: no time range display for T-independent data sets
*    3/16/94 - repaired bug introduced 3/3/94 for EZ data set
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/18/95 - show units in full listing  
* V402: 8/95 - Added support for  LET/D=dset var = ...
*	     - Added missing data flag to outputs
* V420:	10/95 - extended missing flag output to cover dual flags

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, dset
	
* local variable declarations:
	LOGICAL		first_time
	INTEGER 	TM_LENSTR1, ivar, i, dlen, len, grid, taxis,
     .                  slen(4)
	CHARACTER	LEFINT*6, LOHI_STRING*48, TM_FMT*14, line*255
	REAL*8		TM_WORLD, lo, hi
        CHARACTER       rbuff(8)*8

* describe the variables in the set
	risc_buff = ' '
	WRITE ( risc_buff, 3200 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400
           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 400  
* ... output variable code, title and subscript bounds within grid
           DO 100 i = 1, 4
              WRITE( rbuff(i+4),'(I8)' ) ds_grid_start(i,ivar)
              slen(i)  = LOG10( FLOAT(ds_grid_start(i,ivar)) )
              slen(i)  = MIN( 6, 8-slen(i) )   ! last few characters in buffer
              rbuff(i) = LEFINT(ds_grid_end(i,ivar),dlen)
 100       CONTINUE
           line = ds_var_code(ivar)
* ... separate into two lines if variable code is too long (11/92)
           IF ( TM_LENSTR1( line ) .GT. 8 ) THEN
	      risc_buff = ' '
              WRITE ( risc_buff, '(1X,A)' ) ds_var_code(ivar)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
              line = ' '
           ENDIF
	   WRITE ( risc_buff, 4010 )	line,
     .				        ds_var_title(ivar),
     .		(rbuff(i+4)(slen(i):),rbuff(i)(:6),i=1,4)
 4010	   FORMAT( 1X,A8,1X,A31,T42,4(A,':',A) )
 3200	FORMAT(' name',T11,'title',T45,'I',T55,'J',T65,'K',T75,'L')
! in V220: 4010 FORMAT( 1X,A8,1X,A29,T40,3(I5,':',A4),(I5,':',A5) )
* ... write to array first to avoid line wraps due to blanks at end
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

* ... if there is modifying title info give it to
	      IF ( ds_var_titl_mod(ivar) .NE. ' '  ) CALL SPLIT_LIST
     .	(pttmode_explct, lun, '       ('//
     .	ds_var_titl_mod(ivar)(:TM_LENSTR1(ds_var_titl_mod(ivar)))
     .	//')', 0) 

* "FULL" output, only below
	   IF ( full ) THEN
	      grid = ds_grid_number(ivar)
* ... "<units> on grid <name> with 1E35 & -999 for missing data"
	      dlen = TM_LENSTR1( ds_var_units(ivar) )
	      line = '             '//ds_var_units(ivar)(:dlen)//
     .		' on grid '//grid_name(grid)
	      dlen = TM_LENSTR1( line )
	      line = line(:dlen)//' with '//
     .				TM_FMT(ds_bad_flag(ivar),7,14,dlen)
	      dlen = TM_LENSTR1( line )
	      IF ( ds_bad_flag(ivar) .NE. ds_missing_flag(ivar) ) THEN
	         line = line(:dlen)//' & '//
     .				TM_FMT(ds_missing_flag(ivar),7,14,dlen)
	         dlen = TM_LENSTR1( line )
	      ENDIF
	      line = line(:dlen)//' for missing data'
	      CALL SPLIT_LIST(pttmode_explct, lun, line, 0)
* ... world coordinate limits
	      line = ' '
	      len = 1
	      DO 300 i = 1, 3
	         IF ( grid_line(i,grid) .NE. mnormal ) THEN
	            lo = TM_WORLD( ds_grid_start(i,ivar), grid, i, box_lo_lim )
	            hi = TM_WORLD( ds_grid_end  (i,ivar), grid, i, box_hi_lim )
		    line = line(:len) // ww_dim_name(i) // '=' //
     .			LOHI_STRING( lo,hi,i,grid,':',ax_dec_pt(i),dlen )
     .			// '  '
	            len = MIN( 64, len+4+dlen )
	         ENDIF
 300	      CONTINUE
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .		'            '//line(:len), len+12)
	   ENDIF

 400	CONTINUE

* describe the time range (uses "grid" from last variable displayed)
	taxis = ds_time_axis(dset)
        DO 500 grid = 1, max_grids
 500    IF ( grid_line(t_dim,grid) .EQ. taxis ) GOTO 510
        grid = 1                       !  internal error if we get here !!
 510    IF ( full .AND. taxis.NE.mnormal
     .		  .AND. taxis.NE.unspecified_int4) THEN    ! bug fix: 3/94
	   len = line_dim( ds_time_axis(dset) )
	   lo = TM_WORLD( 1,   grid, t_dim, box_middle )
	   hi = TM_WORLD( len, grid, t_dim, box_middle )
	   line = LOHI_STRING( lo,hi,t_dim,grid,' to ',ax_dec_pt(i),dlen )
	   CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			'  time range: '//line(:dlen), 0) 
	ENDIF

* show the data-set-specific user-defined variables
	first_time = .TRUE.
	DO 600 i = 1, max_uvar
	   IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 600
	   IF (uvar_dset(i) .EQ. dset) THEN
	      IF ( first_time ) THEN
	         CALL SPLIT_LIST(pttmode_explct, lun,
     .			' ------------------------------', 31) 
	         first_time = .FALSE.
	      ENDIF
	      CALL  SHOW_1_UVAR(lun, i, line, ' ', full)
	   ENDIF
 600	CONTINUE

* end with a blank line
	CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1) 

	RETURN
	END


