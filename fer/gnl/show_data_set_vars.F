	SUBROUTINE SHOW_DATA_SET_VARS( lun, dset, full, status )

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
* v530  *acm* 3/01 - Show calendar name for alternative calendar types.
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
* V570  *acm* when showing calendar, need  cal_id .LE. max_calendars, not .LT.
* V600  *acm* 1/06  prevent values of LOG10(0.0) 
* V68   *acm* 12/11 Fix ticket 1905. Axis-lengths that need more than 6 characters
*                   were being cut off.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If canceled, the 
*                 various SHOW commnands will not list info for E,F dimensions.
* v6.85  *acm* 4/13 Can show grid for variables in an ensemble aggregated dataset.
* v6.85  *acm* 10/13 See ticket 1098, extraneous lines when error getting grid.
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* v695 *acm*  4/15 Bug fix: output written in loop 300 was cut off
* V698 2/16 *acm*  For ticket 1786: if mode_upcase is canceled write axis name
*                  using original upper/lowercase spelling
* V7   *acm*  6/12 Add Union aggregations: DEFINE DATA/AGG/U, ds_type 'UNI'
* V7    acm  6/16 Ticket 2437: Deprecating mode_6d_lab. 
*		  Adjust output to 6-D or 4-D grids
* V720  3/17 *acm*  clean up long commented-out code referring to aggregations 
*       4/17 *sh* report DSG ragged axis length as "tot 15"
* V720 *acm* 5/17 Implement specific calls in place of CD_GET_VAR_INFO
* V741 *acm* 5/18 Promoting global attributes to a variable in ensembles: list 
*                 the variables synthesized by aggregations with a separator ---
* V743 *acm* 10/10 Fix issue 1897: datasets with E or F axis in any grids gets 6 columns
* V74+ 7/18 *acm* For DSG datasets, if we synthesized a RowSize var, don't show it.
* V751 *acm* 6/19 Remove unused declarations
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings
* V751 7/18 *acm* Call single routine for DSG feature-type names
* V751 8/18 *acm* List the coordinate ranges if DSG dataset
* V76  1/20 *acm* working with Point-type dsg data 
* V76  2/20 *acm* move the code for listin the DSG coordinate ranges
*                 to its own routine, also called by SHOW GRID.
* V76  2/20 *acm* Clean up the SHOW DATA/FULL output for DSG grids -
*                 coordinate info is only at the end of the listing
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ez_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'errmsg.parm'
	include 'xez_info.cmn_text'
	external xez_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xrisc.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
	include 'xpyvar_info.cmn'


* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, dset, status
	
* local variable declarations:

	LOGICAL		FOUR_D_GRID, TM_ITSA_DSG_RAGGED,
     .			first_time, its_delimited, is_undetermined, coordvar,
     .			its_dsg, grid_dsg, first_note, got_it, has_mask
	INTEGER 	TM_LENSTR1, TM_LENSTR, TM_GET_CALENDAR_ID, GRID_FROM_NAME, 
     .			STR_DNCASE,  
     .			ivar, i, dlen, len, grid, taxis, slen(nferdims), tlen, 
     .			nfields, kvar, field_types(maxezvars), 
     .			cal_id, len_cal, j, num_indices, rlen(nferdims), iok,
     .			listdims, idim, igrid, nv, varid, dsg_orient, ndec
	CHARACTER	LEFINT*16, LOHI_STRING*48, TM_FMT*48, 
     .			TM_GET_CAL*32, CD_DSG_FEATURENAME*20,
     .			line*255, cal_name*32
	REAL*8		TM_WORLD, lo, hi
        CHARACTER       rbuff(12)*8, short*6, long*16, delims*128, c*1

* internal parameter declarations:
        CHARACTER*1     tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif


* initialize
	its_delimited = ds_type(dset) .EQ. pds_type_ez
     .            .AND. ds_parm_text(1,dset)(:4) .eq. 'DELI'
	IF (its_delimited) THEN
	   CALL GET_DELIMITED_INFO(nfields, field_types, delims, 
     .				   ds_file_info_ptr(dset) )
	ENDIF

* If the dataset is a DSG dataset, note the FeatureType
        its_dsg = .FALSE.
	DO ivar = 1, maxvars
            IF ( ds_var_setnum(ivar) .NE. dset ) CYCLE
	   igrid = ds_grid_number(ivar)
	   IF (igrid .EQ. int4_init) CYCLE
	   IF (TM_ITSA_DSG_RAGGED(igrid)) THEN
	      its_dsg = .TRUE.
	      EXIT
	   ENDIF
	ENDDO
	IF (its_dsg) THEN

	   dsg_orient = dsg_orientation(dset)

	   risc_buff = '       DSG Feature type '
	   tlen = TM_LENSTR1( risc_buff ) + 1
	   risc_buff = risc_buff(:tlen)//  CD_DSG_FEATURENAME(dsg_orient)

	   tlen = TM_LENSTR1( risc_buff )
	   IF (tlen .GT. 1)  CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
	   
* Is a feature mask set on this dataset?
	   IF (dsg_msk_lm(dset) .NE. int4_init .AND. dsg_msk_lm(dset).NE.unspecified_int4) THEN 
	      risc_buff = 'Mask on features has been set: '
	      tlen = TM_LENSTR1( risc_buff ) + 1
	      
	      CALL GET_DSG_MASK_TITLE (dset, risc_buff(tlen+1:), has_mask, dlen)

	      tlen = TM_LENSTR1( risc_buff ) 
	      IF (dlen .EQ. 0) tlen = tlen - 1

	      IF (tlen .GT. 1)  CALL SPLIT_LIST(pttmode_explct, lun, risc_buff(:tlen), 0) 
	   ENDIF ! feature-mask

	ENDIF

	first_note = .TRUE.

* If mode 6d_lab has been canceled, we will list only the first 4 dimensions.

	listdims = 0
	IF ( .NOT. mode_6d_lab) listdims = 4
	status = ferr_ok

* Adjust number of dimensions. If all grids are 4D, will just show those.

	IF (mode_6d_lab) THEN
	listdims = 0

	DO 80 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 80
           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 80  
* ... output variable code, title and subscript bounds within grid

	   is_undetermined = .TRUE.
	   DO 40 idim = 1, nferdims
	      IF (ds_grid_start(1,ivar) .NE. int4_init) 
     .            is_undetermined = .FALSE. 
  40	   CONTINUE

* Is this an aggregated dataset? - if so then we can go get the grid.
* GRID_FROM_NAME increments the use count for the grid, so call
* DEALLO_DYN_GRID to restore the count where it should be.

           IF (is_undetermined .AND. 
     .         (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT' 
     .          .OR. ds_type(dset).EQ.'UNI')) THEN
              IF (ds_type(dset).EQ.'UNI') THEN
	         igrid = ds_grid_number(ivar)
	      ELSE
	         igrid = GRID_FROM_NAME(ds_var_code(ivar), cx_last, status)
	         IF (status .NE. ferr_ok) GOTO 5000
	         CALL TM_DEALLO_DYN_GRID (igrid)  
	      ENDIF

           ELSE
	      igrid = ds_grid_number(ivar)
           ENDIF

           IF (.NOT.is_undetermined) THEN	
	      IF ( FOUR_D_GRID(igrid)) THEN
	         listdims = MAX(listdims,4)
	      ELSE
	         listdims = 6
	      ENDIF
           ENDIF
 80	CONTINUE

	IF (listdims .EQ. 0) listdims = nferdims
	ENDIF ! adjusting listdims

* describe the variables in the set

	risc_buff = ' '
        IF (listdims .EQ. nferdims) THEN
           WRITE ( risc_buff, 3200 )
        ELSE
           WRITE ( risc_buff, 3210 )
        ENDIF

	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 


	kvar = 0
	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400
	   kvar = kvar + 1
           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 400  

	   IF (its_dsg .AND. ds_var_code(ivar) .EQ. 'ferRowSize') GOTO 400
	   
* ... output variable code, title and subscript bounds within grid

	   is_undetermined = .TRUE.
	   DO 200 idim = 1, nferdims
	      IF (ds_grid_start(1,ivar) .NE. int4_init) 
     .            is_undetermined = .FALSE. 
  200	   CONTINUE

* Is this an aggregated dataset? - if so then we can go get the grid.
* GRID_FROM_NAME increments the use count for the grid, so call
* DEALLO_DYN_GRID to restore the count where it should be.

           IF (is_undetermined .AND. 
     .         (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT' 
     .          .OR. ds_type(dset).EQ.'UNI')) THEN
              IF (ds_type(dset).EQ.'UNI') THEN
	         igrid = ds_grid_number(ivar)
	      ELSE
	         igrid = GRID_FROM_NAME(ds_var_code(ivar), cx_last, status)
	         IF (status .NE. ferr_ok) GOTO 5000
	         CALL TM_DEALLO_DYN_GRID (igrid)  

*  Fill the grid in as in define_ef_aggregate_grid

	         IF (igrid .NE. unspecified_int4) THEN
	            DO 220 nv = 1, maxvars
	               IF (ds_grid_number(nv) .EQ. igrid) EXIT ! loop
 220	            CONTINUE
	         ENDIF

	      ENDIF

	      IF (nv .LE. maxvars) THEN
	         DO 230 idim = 1, nferdims
	            ds_grid_start(idim,ivar)= ds_grid_start(idim,nv)
	            ds_grid_end(idim,ivar)  = ds_grid_end(idim,nv)
	            ds_ordering(idim,ivar)  = ds_ordering(idim,nv)
 230	         CONTINUE
	         is_undetermined = .FALSE.
              ENDIF
           ELSE
	      igrid = ds_grid_number(ivar)
           ENDIF

           IF (.NOT.is_undetermined) THEN
	      
	      grid_dsg = TM_ITSA_DSG_RAGGED(igrid)
	      IF (grid_dsg) THEN
	         dsg_orient = dsg_orientation(dset)
		 IF (dsg_orient.EQ.pfeatureType_TrajectoryProfile .OR.
     .		     dsg_orient.EQ.pfeatureType_TimeseriesProfile)
     .		   dsg_orient = z_dim
	      ENDIF

              DO 100 i = 1, listdims
                 WRITE( rbuff(i+listdims),'(I8)' ) ds_grid_start(i,ivar)
                 IF (FLOAT(ds_grid_start(i,ivar)) .EQ. 0.0) THEN
                    slen(i) = 1
                 ELSE
                    slen(i)  = LOG10( FLOAT(ds_grid_start(i,ivar)) )
                 ENDIF
                 slen(i)  = MIN( 6, 8-slen(i) )   ! last few characters in buffer
                 long = LEFINT(ds_grid_end(i,ivar),dlen)
                 rlen(i) = dlen
                 short = LEFINT(ds_grid_end(i,ivar),dlen)
                 rbuff(i) = short
 100          CONTINUE
           ENDIF

* List synthesized variable in aggregations after a separator

	   IF ( (ds_var_title(ivar)(1:12) .EQ. 'Global Att *' .OR.
     .	         ds_var_title(ivar)(1:20) .EQ. 'Member dataset names')
     .	         .AND. first_note) THEN
	      line = ' ------------------------------'
	      CALL SPLIT_LIST(pttmode_explct, lun, line, 0) 
	      first_note = .FALSE.
           ENDIF

	   line = ds_var_code(ivar)
	   IF (.NOT.mode_upcase_output) THEN  ! get original variable name
	      CALL CD_GET_VAR_ID (dset, ds_var_code(ivar), varid, status)
	      CALL CD_GET_VAR_VARNAME (dset, varid, line, status)
	   ENDIF

* ... separate into two lines if variable code is too long (11/92)
           IF ( TM_LENSTR1( line ) .GT. 8 ) THEN
	      risc_buff = ' '
              WRITE ( risc_buff, '(1X,A)' ) line
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
              line = ' '
           ENDIF
	   IF (is_undetermined) THEN 
	      WRITE ( risc_buff, 4011 )	line,
     .				        ds_var_title(ivar),
     .				        '... Aggregate grid undetermined' 
           ELSE
	      WRITE ( risc_buff, 4010 )	line,
     .				        ds_var_title(ivar),
     .	     (rbuff(i+listdims)(slen(i):),rbuff(i)(:6),i=1,listdims)

 4010	      FORMAT( 1X,A8,1X,A31,T42,6(A,':',A) )
 4011	      FORMAT( 1X,A8,1X,A31,T42,A )
* *kob* 3/99 check for NORMAL Axis and insert ... rather than 1:1 
              DO 150 i = 1, listdims
	         IF ( grid_line(i,ds_grid_number(ivar)) .EQ. mnormal) THEN
		    tlen = 34 + (i*10)
		    risc_buff(tlen:tlen+2) = '...'
	         ENDIF
 150	      CONTINUE
* *sh* 4/17 a DSG obs axis actually shows the total of feature lengths
	      IF (grid_dsg) THEN
	         tlen = 34 + (dsg_orient*10) - 2
	         risc_buff(tlen:tlen+4+rlen(dsg_orient))
     .			= 'tot '//rbuff(dsg_orient)(:rlen(dsg_orient))
	      ENDIF

           ENDIF

* See if any axis lengths were too long for the standard formatting.
* Do a more brute-force formatting which may or may not line up 
* all variables perfectly

           iok = 1
           DO 160 i = 1, listdims
	      IF (rlen(i) .GT. 6) iok = 0
 160	   CONTINUE

	   IF (iok .EQ. 0) THEN  ! re-format the line for this variable

* List synthesized variable in aggregations after a separator

           IF ( (ds_var_title(ivar)(1:12) .EQ. 'Global Att *'.OR.
     .	         ds_var_title(ivar)(1:20) .EQ. 'Member dataset names')
     .	       .AND. first_note) THEN
	      line = ' ------------------------------'
	      CALL SPLIT_LIST(pttmode_explct, lun, line, 0) 
	      first_note = .FALSE.
           ENDIF

	   line = ds_var_code(ivar)
	   IF (.NOT.mode_upcase_output) THEN  ! get original variable name
	      CALL CD_GET_VAR_ID (dset, ds_var_code(ivar), varid, status)
	      CALL CD_GET_VAR_VARNAME (dset, varid, line, status)
	   ENDIF
* ... separate into two lines if variable code is too long (11/92)
           IF ( TM_LENSTR1( line ) .GT. 8 ) THEN
	      risc_buff = ' '
              WRITE ( risc_buff, '(1X,A)' ) line
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
              line = ' '
           ENDIF

	   WRITE ( risc_buff, 4010 )	line,
     .				        ds_var_title(ivar)
           tlen = MAX(40, TM_LENSTR1(risc_buff))
	   risc_buff(tlen+1:tlen+2) = '  ' 
           tlen = tlen + 2

           DO 170 i = 1, listdims
	      IF ( grid_line(i,ds_grid_number(ivar)) .EQ. mnormal) THEN
		 risc_buff(tlen:tlen+8) = '   ...   '
		 tlen = tlen + 9
              ELSE
                 long = LEFINT(ds_grid_start(i,ivar),dlen)
	         risc_buff(tlen+1:tlen+dlen) = long(1:dlen)
	         tlen = TM_LENSTR1(risc_buff)
	         risc_buff(tlen+1:tlen+1) = ":"
	         tlen = tlen + 1
                 long = LEFINT(ds_grid_end(i,ivar),dlen)
	         risc_buff(tlen+1:tlen+dlen) = long(1:dlen)
	         tlen = TM_LENSTR1(risc_buff)
	         risc_buff(tlen+1:tlen+2) = '  ' 
	         tlen = tlen + 2

	      ENDIF
 170	   CONTINUE
	   ENDIF ! iok

 3200	FORMAT(' name',T11,'title',T45,'I',T55,'J',T65,'K',T75,'L',T85,'M',T95,'N')
 3210	FORMAT(' name',T11,'title',T45,'I',T55,'J',T65,'K',T75,'L')

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
*             Appending space just to get rid of a nonsense compiler warning
	      risc_buff = line(:dlen)//' '
	      len_rbuff = dlen
	      line = risc_buff(:len_rbuff)//' with '//
#else
	      line = line(:dlen)//' with '//
#endif
     .				TM_FMT(ds_bad_flag(ivar),7,14,dlen)
	      dlen = TM_LENSTR1( line )
	      IF ( ds_bad_flag(ivar) .NE. ds_missing_flag(ivar) ) THEN
#ifdef NO_PREPEND_STRING
*             Appending space just to get rid of a nonsense compiler warning
	      risc_buff = line(:dlen)//' '
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
	      DO 300 i = 1, listdims
	         IF (its_dsg) CYCLE  ! DSG coords summarized later.
	         IF ( grid_line(i,grid) .NE. mnormal ) THEN

		    IF (i .NE. t_dim) THEN  ! Skip time dim

	               lo = TM_WORLD( ds_grid_start(i,ivar), grid, i, box_lo_lim )
	               hi = TM_WORLD( ds_grid_end  (i,ivar), grid, i, box_hi_lim )

		       ndec = ax_dec_pt(i)

		       ndec = -5  ! 
		       IF (i .EQ. y_dim) ndec = -3
#ifdef NO_PREPEND_STRING
*                      Appending space just to get rid of a nonsense compiler warning
	               risc_buff = line(:len)//' '
 	               len_rbuff = len
		       line = risc_buff(:len_rbuff)// ww_dim_name(i) // '=' //
#else
		       line = line(:len) // ww_dim_name(i) // '=' //
#endif
     .			 LOHI_STRING( lo,hi,i,grid,':',ndec,dlen )
     .			 // '  '
	               len = MIN( 128, len+4+dlen )
	            ENDIF ! Skip time dim

	         ENDIF
 300	      CONTINUE
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .		'            '//line(:len), len+12)
	   ENDIF

 400	CONTINUE

* If the dataset is a DSG dataset, show the x, y, z, t range for the 
* observation variables.

	IF ( full  .AND. grid_dsg) CALL SHOW_DSG_RANGES ( dset, .FALSE., lun )

* describe the time range (uses "grid" from last variable displayed)
	taxis = ds_time_axis(dset)
c	IF ( full  .AND. grid_dsg) taxis = grid_line(t_dim, dsg_xlate_grid(dset))

* ... check among the static grids
        DO grid = 1, max_grids
           IF ( grid_line(t_dim,grid) .EQ. taxis ) GOTO 520
 	ENDDO
* ... check among the dynamic grids
	grid = 0
 510	CALL TM_NEXT_DYN_GRID( grid, *515)
	  IF ( grid_line(t_dim,grid) .EQ. taxis ) GOTO 520
	GOTO 510
 515	grid = 1		!  internal error if we get here !!
 520    IF ((.NOT.grid_dsg) .AND. full .AND. taxis.NE.mnormal
     .		  .AND. taxis.NE.unspecified_int4) THEN    ! bug fix: 3/94
	   len = line_dim( taxis )
	   lo = TM_WORLD( 1,   grid, t_dim, box_middle )
	   hi = TM_WORLD( len, grid, t_dim, box_middle )
	   line = LOHI_STRING( lo,hi,t_dim,grid,' to ',ax_dec_pt(t_dim),dlen )
	   IF (.NOT. grid_dsg) CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)

* calendar name?  only if not Gregorian.
	   cal_name = TM_GET_CAL (grid)
	   cal_id = TM_GET_CALENDAR_ID (cal_name)
	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      len_cal = TM_LENSTR(cal_name)
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .	       '  time range: '//line(:dlen)//
     .	       ' Calendar:'//cal_name(:len_cal), 0) 
	   ELSE
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .			'  time range: '//line(:dlen), 0) 
	   ENDIF
	ENDIF

* show the data-set-specific pyvar (cat_pystat_var) variables
        first_time = .TRUE.
        DO 580 i = 1,maxpyvars
            IF ( pyvar_ndarray_obj(i) .EQ. 0 ) GOTO 580
            IF ( pyvar_dset_number(i) .NE. dset ) GOTO 580
            IF ( first_time ) THEN
                CALL SPLIT_LIST(pttmode_explct, lun,
     .                          ' ------ Python Variables ------', 31) 
                first_time = .FALSE.
            ENDIF
            CALL SHOW_1_PYVAR(lun, i, full)
 580    CONTINUE


* show the data-set-specific user-defined variables
* Skip intermediate variables by checking uvar_parent

	first_time = .TRUE.
*	DO 600 i = 1, max_uvar
*	   IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 600
        CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
        DO 600 j = 1, num_indices
           i = deleted_list_result(j)
	   IF (uvar_dset(i) .EQ. dset .AND.
     .         uvar_parent(i) .EQ. 0   ) THEN
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
	      WRITE (line(len-2:len),'(I3.3)') ICHAR(c) 
	    ENDIF
	    len = len + 1
	    line(len:len) = ','
	  GOTO 700
 710	  CALL SPLIT_LIST(pttmode_explct, lun,
     .			  ' /DELIMITERS="'//line(:len)//'"', len+15) 
	ENDIF

5000    RETURN
5100	CALL ERRMSG( ferr_invalid_command, status,
     .		     'variable has more than 4 dimensions. Set mode 6D_LAB'
     .		     //pCR//'Output will automatically adjust to 4D or 6D grid'
     .		     //pCR//cmnd_buff(:len_cmnd),
     .			*5000 )
	END
