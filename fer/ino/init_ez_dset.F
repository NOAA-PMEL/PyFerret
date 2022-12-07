	SUBROUTINE INIT_EZ_DSET( dset_name, dset, status )

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
* initialize a data set of type "EZ" (simple ASCII or unformatted file with
* no attendant descriptor

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/20/89
*	  2/2/90 - make sure its really an EZ set
*	  2/8/90 - default /VAR is "V1"
*	 3/27/90 - cancel data set (vs warning) if /VAR or /COL is changed
*	  4/2/90 - cancel data set (vs warning) if already_init
* Unix/RISC port - filesystem specific needs - new routine to evaluate path
*                - and longer saved descriptor name
* V3.1      5/94 - added /ORDER= 
* V320  11/94 - *sh* IBM port - need to pass 4 character blank in TM_INQ_PATH
*       11/94 - title needs to be case sensitive
* V420 (V411) 10/95 - for IBM port change "READ(dset_name,*" to use '(I)'
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.
* V450 7/97 - Increase array var_list holding dset var names
*      7/98 *js*   Mods to support new /swap and /type qualifiers for stream
*                  datasets
* 3/99 *kob* - nag F90 wants to use eqv for logical expressions
* 5/99 *sh* - added arguments to EQUAL_PERMUTE call
* V510 *sh* 3/00 - increased allowable filename length
*	    4/00 - bump use counts on grids and axes used by this data set
* V530 *sh* 10/00 - added /FORMAT=delimited/TYPE=data_types
* V533 *sh* 7/01 - bug fix: filenames beginning with digits
* V540 *sh* 10/01 - bug fig -- process TMAP error from ez_init_dset
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
* v604 *acm* 7/07  increase length of ds_name
* v614 *acm* 11/08 robust fix for bug 287: filenames beginning with digits. 
*                  Check if the filename contains characters other than digits.
* V64  10/09 *acm* longer string length for grid_name
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*       *acm* 3/12 cleanup old VMS ifdef
* V687  2.14 *acm* fix ticket 2144: Make buffer for list of varnames longer
* V692  7/14 *acm* fix ticket 2182: For ez datasets opened w/o a grid spec, 
*                  count the records and define an axis
* V701  7/16 *acm* Ticket 2454. If the user gives lower-case names when opening
*                  an EZ dataset, keep those names on output, etc.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations
	INTEGER		dset, status, i
	CHARACTER	dset_name*(*)

* internal variable declarations:
	INTEGER		nferdimsp1 
	PARAMETER	(nferdimsp1 = nferdims+1)
	LOGICAL		TM_DIGIT, VALID_GRID, its_delimited,
     .			have_mods, already_init, by_number, swap
	LOGICAL         have_type
	INTEGER		TM_LENSTR1, TM_DSET_NUMBER, GRID_FROM_NAME,
     .			tmap_status, loc, skip, grid, columns,
     .			order(nferdimsp1), idim, s1, slen, ndigits, nread,
     .                  ndims, nvars, ngatts, recdim
	CHARACTER	TM_INQ_PATH*2048, TM_FMT*48, var_list*2048,
     .			path_name*2048, ezform*80, title*128, grid_name*64,
     .			type*2048, ctype*16, delim*24, buff*20
	REAL		rval

* local parameter declarations
	CHARACTER*1	tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

* functions
	INTEGER         br_set_type
* initialize
	have_mods = .FALSE.
	swap = .FALSE.
	have_type = .FALSE.

* interpret the data set name or number given
* 3 possibilities:
*   i) dset implied from current default
*  ii) dset given by number
* iii) dset given by name  
	IF ( dset_name .EQ. ' ' ) THEN
	   by_number = .TRUE.
	   already_init = .TRUE.
	   dset = cx_data_set(cx_last)
	   GOTO 200
	ELSEIF ( TM_DIGIT(dset_name) ) THEN
* *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	   READ (dset_name,*,ERR=100) dset	! 10/95
#else
	   READ (dset_name,'(I)',ERR=100) dset	! 10/95
#endif

* Is this a dataset name that starts with a number ?
* (some compilers get an error on the above read if
*  dset_namecontains other than digits, some do not.)

	   slen = TM_LENSTR1(dset_name)
	   ndigits = 1
	   IF (dset .GE. 10) ndigits = 2
	   IF (slen .GT. ndigits) THEN
              s1 = ndigits + 1
              DO i = s1,slen
	         IF (.NOT. TM_DIGIT(dset_name(i:i)) ) GOTO 100
              ENDDO
           ENDIF
	      
	   by_number = .TRUE.
	   already_init = .TRUE.
	   GOTO 200
	ENDIF

* open data set by name
 100	by_number = .FALSE.

        path_name = TM_INQ_PATH( dset_name, 'FER_DATA', '    ',
     .                                   .TRUE., TMAP_status )
	IF ( TMAP_status .EQ. merr_notmounted ) GOTO 5400
	IF ( TMAP_status .NE. merr_ok ) GOTO 5900
	dset = TM_DSET_NUMBER( path_name )
	already_init = dset .NE. unspecified_int4

 200	IF ( by_number ) THEN
	   IF   ( dset .LT. 1
     .	   .OR.   dset .GT. maxdsets
     .	   .OR.   dset .GT. max_gfdl_dsets )		GOTO 5300
	   IF ( ds_name( dset ) .EQ. char_init2048 )	GOTO 5300
	ENDIF

* already initialized ? - cancel it
	IF ( already_init ) THEN
* ... is it an EZ set ?
	   IF ( ds_type(dset) .NE. '  EZ' ) GOTO 5100
	   CALL PURGE_DSET( dset )	! remove memory-resident variables
	   CALL PURGE_ALL_UVARS		! delete (possibly dependent) uvars
* (note: this purge is unnecessarily strict - could be refined in the future)

	   CALL CANCEL_DATA_SET( dset, status )
	
	   IF ( status .NE. ferr_ok ) RETURN
	   already_init = .FALSE.
	ENDIF

* get number of columns if specified
	loc = qual_given( slash_set_data_column )
	IF ( loc .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   rval, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   columns = INT( rval )
	ELSE
	   columns = int4_init
	ENDIF

* read the file format
        loc = qual_given( slash_set_data_form )
        IF ( loc .GT. 0 ) THEN
           have_mods = .TRUE.
           CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        ezform, status )
           IF ( status .NE. ferr_ok ) RETURN
	   its_delimited = ezform(1:4) .EQ. 'DELI'
        ELSE
           ezform = char_init16
	   its_delimited = .FALSE.
        ENDIF

* the /TYPE qualifier may refer to either delimited reads or binary reads
* Note: "?TYPE" is overloaded -- used independently for STREAM and
* DELIMITED format reads
        loc = qual_given( slash_set_data_type )
	IF ( loc .EQ. 0 ) THEN
	   IF (its_delimited) THEN
	      type = ' '  ! to be determined in ez_init_dset
	   ELSE
#ifdef double_p
	      type = 'R8'  ! default for stream reads
#else
	      type = 'R4'  ! default for stream reads
#endif
	   ENDIF
	ELSE
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      type, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF (.NOT.its_delimited) have_type = .TRUE. ! Joe's binary stream
	ENDIF

* /DELIMIT=
	loc = qual_given( slash_set_data_delimiters )
	IF ( loc .GT. 0 ) THEN
	   IF (.NOT.its_delimited) GOTO 5450
	   CALL EQUAL_DELIMIT(
     .			cmnd_buff(qual_start(loc):qual_end(loc)),
     .			delim, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   delim = tab//','//CHAR(0)  ! NULL terminated
	ENDIF

* get list of variable names: SET DATA/EZ/VAR="..." name 
	loc = qual_given( slash_set_data_var )
	IF ( loc .EQ. 0 ) THEN
	   var_list = ' '
	ELSE
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      var_list, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* /SKIP
	loc = qual_given( slash_set_data_skip )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   rval, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   skip = INT( rval )
	ELSE
	   skip = 0  ! was int4_init pre 10/00
	ENDIF

	CALL EZ_INIT_DSET( path_name, ezform, var_list, type, delim,
     .			   skip, columns, dset, tmap_status )
	IF ( tmap_status .NE. merr_ok ) GOTO 5900

* always process any other qualifiers given
	loc = qual_given( slash_set_data_title )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      title, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   title = char_init80
	ENDIF
	loc = qual_given( slash_set_data_order )    ! 5/94
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_PERMUTE( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                         .TRUE., .TRUE., .FALSE., order, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   DO idim = 1,nferdimsp1
	      order(idim) = idim
 	   ENDDO
	ENDIF
	loc = qual_given( slash_set_data_grid )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      grid_name, status )
           IF ( status .NE. ferr_ok ) RETURN
	   grid = GRID_FROM_NAME( grid_name, cx_last, status )
           IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.VALID_GRID( grid ) ) GOTO 5500
	ELSEIF ( .NOT.already_init ) THEN

* ... insert the default defining grid for STREAM
	   if (ezform(:3) .EQ. 'STR') then
	      have_mods = .TRUE.
	      grid = mgrid_ez_dflt
	   ELSE

* ... For ascii and unformatted sets, get the datset count and define a 
*     basic x axis, not the ez abstract axis
	      CALL EZ_COUNT_DSET ( dset, ezform, nread, status )
              IF ( status .NE. ferr_ok ) RETURN
	      nread = nread - skip

	      IF (columns .GT. 1) THEN
	         CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts,
     .            recdim, status)
                 nread = nread*columns/nvars
	      ENDIF

	      buff = TM_FMT(DBLE(nread), 0, 12, slen)
	      grid_name = 'X[GX=1:'//buff(:slen)//':1]'
	      grid = GRID_FROM_NAME( grid_name, cx_last, status )
              IF ( status .NE. ferr_ok ) RETURN
	      IF ( .NOT.VALID_GRID( grid ) ) GOTO 5500
	      have_mods = .TRUE.
	   endif
	ELSE
	   grid = int4_init
	ENDIF

c       /swap
	loc = qual_given(slash_set_data_swap)
	if (loc .GT. 0) then
	   swap = .TRUE.
	END IF
	   
	IF ( have_mods ) THEN
	   CALL EZ_MOD_DSET( dset, title, ezform, grid, skip, order,
     .				tmap_status )
	   IF ( tmap_status .NE. merr_ok ) GOTO 5900
	ENDIF

* Make sure no v permutations are defined for non-stream formats,
* and don't allow /swap or /type flags 
	if (ezform(:3) .NE. 'STR') then
	   do i=1,nferdims
	      if (order(i) .eq. nferdimsp1) goto 5700
	   end do
* 3/99 *kob* - nag F90 wants to use eqv for logical expressions
	   if (swap .eqv. .TRUE. .or. have_type .eqv. .TRUE.) then
	      goto 5800
	   end if
	else
	   call toChar(type, ctype)
	   if (BR_SET_TYPE(ctype) .EQ. 0) goto 5600
	   call BR_SET_ATTS(order, swap)
	end if


* bump use counts on grids and axes used by this data set
	CALL TM_DSET_USE_GRIDS( dset )

* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5050	CALL ERRMSG( ferr_erreq, status, dset_name, *5000 )
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		ds_name(dset)(:TM_LENSTR1(ds_name(dset)))//
     .		' is not an EZ data set', *5000 )
 5300	CALL ERRMSG( ferr_unknown_data_set, status,
     .		'invalid data set number for SET DATA/EZ', *5000 )
 5400	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5450	CALL ERRMSG(ferr_invalid_command, status,
     .        '/DELIMIT only allowed for DELIMITED format', 
     .        *5000)
 5500	CALL ERRMSG( ferr_grid_definition, status, grid_name, *5000 )
 5600	call errmsg(ferr_invalid_command, status,
     .        'invalid argument for /TYPE', *5000)
 5700	CALL ERRMSG(ferr_invalid_command, status,
     .        'variable permutations only allowed for STREAM format', 
     .        *5000)
 5800	CALL ERRMSG(ferr_invalid_command, status,
     .        '/SWAP or /TYPE only allowed for STREAM format', 
     .        *5000)
 5900	CALL ERRMSG ( ferr_TMAP_error, status, ' ', *5000 ) 
	END
