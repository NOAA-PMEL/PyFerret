	SUBROUTINE SHOW_DATA_SET_VARS( lun, dset, full )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* V500: 3/99 *kob* - added check for NORMAL Axes - output only ... rather
*                    than a 1:1 or 100:100
*       4/99 *kob* - bug fix to above - check grid_line to see if axis is
*                    truly normal
* V510: 4/00 *sh* - search dynamic grids for netCDF time axis
* V530: 10/00 *sh* - added information about data typed

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ez_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xez_info.cmn_text'
	external xez_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xrisc.cmn'

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, dset
	
* local variable declarations:
	LOGICAL		first_time, its_delimited
	INTEGER 	TM_LENSTR1, ivar, i, dlen, len, grid, taxis,
     .                  slen(4), tlen, nfields, kvar,
     .			field_types(maxezvars)
	CHARACTER	LEFINT*6, LOHI_STRING*48, TM_FMT*14, line*255
	REAL*8		TM_WORLD, lo, hi
        CHARACTER       rbuff(8)*8, delims*128, c*1

* internal parameter declarations:
        CHARACTER*1     tab
        PARAMETER     ( tab = CHAR(9) )

* initiailze
	its_delimited = ds_type(dset) .EQ. pds_type_ez
     .            .AND. ds_parm_text(1,dset)(:4) .eq. 'DELI'

	IF (its_delimited) THEN
	   CALL GET_DELIMITED_INFO(nfields, field_types, delims, 
     .				   ds_file_info_ptr(dset) )
	ENDIF

* describe the variables in the set
	risc_buff = ' '
	WRITE ( risc_buff, 3200 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
	kvar = 0
	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400
	   kvar = kvar + 1
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
* *kob* 3/99 check for NORMAL Axis and insert ... rather than 1:1 
           DO 150 i = 1, 4
	      IF ( grid_line(i,ds_grid_number(ivar)) .EQ. mnormal) THEN
		 tlen = 34 + (i*10)
		 risc_buff(tlen:tlen+2) = '...'
	      ENDIF
 150	   CONTINUE
	   
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

* ... data type if interesting
	      IF (ds_var_type(ivar) .EQ. ptype_string ) THEN
	        CALL SPLIT_LIST(pttmode_explct, lun,
     .		'            '//'Data type = TEXT', 28)
	      ELSEIF (its_delimited                   ) THEN
	        CALL SPLIT_LIST(pttmode_explct, lun,
     .		'            '//'Data type = NUMERIC', 31)
	      ENDIF

* ... field type conversion
	      IF (its_delimited) THEN
	        CALL SPLIT_LIST(pttmode_explct, lun,
     .		'            '//'Conversion /TYPE='//
     .			     ftyp_name(field_types(kvar)), 0)
	      ENDIF

* ... "<units> on grid <name> with 1E35 & -999 for missing data"
	      dlen = TM_LENSTR1( ds_var_units(ivar) )
	      line = '             '//ds_var_units(ivar)(:dlen)//
     .		' on grid '//grid_name(grid)
	      dlen = TM_LENSTR1( line )
#ifdef NO_PREPEND_STRING
	      risc_buff = line(:dlen)
	      len_rbuff = dlen
	      line = risc_buff(:len_rbuff)//' with '//
#else
	      line = line(:dlen)//' with '//
#endif
     .				TM_FMT(ds_bad_flag(ivar),7,14,dlen)
	      dlen = TM_LENSTR1( line )
	      IF ( ds_bad_flag(ivar) .NE. ds_missing_flag(ivar) ) THEN
#ifdef NO_PREPEND_STRING
	      risc_buff = line(:dlen)
	      len_rbuff = dlen
	      line = risc_buff(:len_rbuff)//' & '//
#else
	         line = line(:dlen)//' & '//
#endif
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
#ifdef NO_PREPEND_STRING
	            risc_buff = line(:len)
 	            len_rbuff = len
		    line = risc_buff(:len_rbuff)// ww_dim_name(i) // '=' //
#else
		    line = line(:len) // ww_dim_name(i) // '=' //
#endif
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
* ... check among the static grids
        DO 500 grid = 1, max_grids
 500    IF ( grid_line(t_dim,grid) .EQ. taxis ) GOTO 520
* ... check among the dynamic grids
	grid = 0
 510	CALL TM_NEXT_DYN_GRID( grid, *515)
	  IF ( grid_line(t_dim,grid) .EQ. taxis ) GOTO 520
	GOTO 510
 515	grid = 1		!  internal error if we get here !!
 520    IF ( full .AND. taxis.NE.mnormal
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

* show delimiters used for delimited files
	IF (full .AND. its_delimited) THEN
* ... convert delimiter string into readabl;e form
	  line = ' '
	  len = 0
* ... a DO-WHILE loop until the null is encountered
	  i = 0
 700	  i = i + 1
	    c = delims(i:i)
	    IF (c.EQ.CHAR(0)) THEN
	      len = len - 1   ! remove final comma
	      GOTO 710
	    ELSEIF (c.EQ.tab) THEN
	      len = len + 2
#ifdef NO_DOUBLE_ESCAPE_SLASH
              line(len-1:len) = '\'//'t'
#else
              line(len-1:len) = '\\'//'t'
#endif	     
	    ELSEIF (c.EQ.',') THEN
	      len = len + 2
#ifdef NO_DOUBLE_ESCAPE_SLASH
              line(len-1:len) = '\'//','
#else
              line(len-1:len) = '\\'//','
#endif	     
	    ELSEIF ( (c.GE.' ' .AND. c.LE.'~') )THEN
	      len = len + 1
	      line(len:len) = c  ! printable character
	    ELSE
	      len = len + 3
	      WRITE (line(len-2:len),'(I3.3)') INT(c) 
	    ENDIF
	    len = len + 1
	    line(len:len) = ','
	  GOTO 700
 710	  CALL SPLIT_LIST(pttmode_explct, lun,
     .			  ' /DELIMITERS="'//line(:len)//'"', len+15) 
	ENDIF

	RETURN
	END
