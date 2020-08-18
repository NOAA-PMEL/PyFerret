	SUBROUTINE XEQ_LIST

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
* execute the LIST command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/16/86
* revision 0.1 - 1/5/87	 - changes to accomodate multiple mrs in expression
* revision 0.2 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.3 - 8/29/87 - changes for unformatted LIST: OPEN file in this rout.
* revision 0.4 - 9/9/87  - added DSF_LIST call (Denbo's BIBO output)
* revision 0.5 - 9/16/87 - added want_cmnd_data to CALL GET_CMND_DATA
* revision 0.6 - 9/26/87 - added AUTO filename option and headers automatically
*			   for /APPEND when file doesn't exist
* revision 0.7 - 10/5/87 - revamped SET LIST logicals
* revision 0.8 - 3/7/88  - added list_TMAP
* revision 0.9 - 6/24/88 - use LIST/FILE[/APPEND] to direct output to file
*			 - allow multiple variables in a /FORMAT=GT list
* revision 1.0 - 2/22/89 - check for interrupts after each variable
* V200:  5/18/89 - minor changes for 4D-symmetrical
*		 - context passed in is_cx instead of is_big_cx
*		 - list_unformatted and list_format_given are cancelled at end
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
*		 - permutable listing order
*	12/19/89 - corrected bug in LIST/FORM=TMAP
*	 3/20/90 - eliminated forced heading for /APPEND when file doesn't exist
*	 4/27/90 - unformatted output types without a /FILE generate an error
*	 5/15/90 - DSF (BIBO) output neede to call AUTO_FILENAME, too
*	  7/2/90 - allow multiple variables per line, numerous style changes
*		 - interrupt checking inside list routines
*	 7/31/90 - added EPIC format and MULTI lists use work arrays
* Unix/RISC port - changed OPEN syntax for Unix filesystem
*                - changed /FILE= to preserve lower case
* SUN port 3/92 *sh* - unformatted record length in bytes
*                    - discovered "perm" is unitialized if not specified
* V230:  6/9/92 - added netCDF output including changes to the use of
*                 XPROG_STATE variables dealing w/ format type
*                 SET LIST is now effective only for the NEXT list
*       7/27/92 - don't remove protections from "command data" until AFTER
*                 listing because permutation requires memory management
*        8/6/92 - CDF was using is_mr instead of mr_perm (minimum reqsted reg.)
*           and - error during GET_PROT_CMND_DATA left /FORMAT= set
*      10/14/92 - bug fix: wrong GOTO following EQUAL_FORMAT error
* V300:    5/93 - error message bug: filename truncated
*	  10/93 - ALPHA port - explicit error message if file exists because
*		  OSF/1 error message is "Bad magic number"
* V314:	8/24/94 *kob* IBM port - RECORDTYPE not allowed in open statments under
*				 AIX fortran.  Add ifdef 1.
*	11/3/94 *sh* IBM port - ACCESS=APPEND isn't allowed but STATUS=OLD
*				defaults to this behavior
*	11/5/94 *sh* IBM port - disable GT format output on IBM.  Unable to
*			append on unformatted file (not checked thoroughly)
* V320:  5/94 - array "memory" as a calling argument
*	10/94 - added code for /HEADING=enhanced
*	11/94 - trap LIST/FORMAT without /FILE in GUI mode as an error
*	11/94 - added SAVE/HEADING="title"
*	12/94 - added /FORMAT=stream
*    12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*     1/17/95 - increased output filename length from 100 characters to 200
*     1/25/95 - added /TITLE= and /HEADING=enhanced (and unenhanced) for netCDF
* V420   9/95 - added /FORMAT=comma and tab delimited
* Linux Port 5/97 *kob* - had to add ifdef F90_OPEN_FILE_APPEND because f90
*			  uses "POSITION" instead of "ACCESS" to specify 
*			  opening a file for append.  I also had to create a
*			  new variable called "fposition", which stores the 
*			  open position of the file, either "append" or "asis".
*			  this had to be done in order for a new file to have 
*			  a valid "position" for f90 opens.
* V430	 7/96 - *kob* reclen_in_bytes not appropriate for sgi..added 
*		      "ifndef sgi" to prevent 1 being set
* V450	11/96 - added LIST/CLOBBER
*		added pcdf_mode_cache for DODS caching
*       7/98 *js* Changed size of perm to 5 to support parsing of 
*                 xyztv permutations (unsupported for now)
*      12/98 *js* Disable listing to files if in secure mode
* V500  2/99  - *kob* added LIST/QUIET
*	5/99 - *sh- added arguments to EQUAL_PERMUTE
* V530  8/00 *sh* - bug fix of /CLOBBER with /FORM=STREAM
*		  - set reclen correctly for string output
*       *sh* 9/00  - added initial data type support in get_cmnd_data 
*	*sh* 11/00 - fixed bug: permutation for all vars written was determined
*		     from context of first var, only
*       *sh8 1/01  - bug fix to 11/00 bug fix!
* v542  *kob* 10/02 - add F90_OPEN_FILE_CLOBBER option to make sure that the F90 
*                     OPEN POSITION directive is not blank when opening an 
*                     existing file for rewrite.  This causes an F90 OPEN error
* V554 *acm* 9/03  Add output in /DODS binary format for list/format=DODS, 
*                  see the flag do_dods
* V581 *acm* 3/05  Increase length of string buff so that title can be longer.
* V600: *acm* 8/05 - make multi-variable tab and comma-delimited listings of
*                    multiple variables like free-format listings, rather than 
*                    like LIST/SINGLY. (bug 1273)
* V600  *acm* 4/06 change how mr_perm is passed to LIST_CMND_DATA,due to 64-bit problems  
* V602  *acm* 1/07 Use risc_buff rather than the local variable buff for reading inputs 
*                  and for the title - fixing bug 1482
* V604  *acm* 7/07 Dataset names have been increased to 1024, but on output this runs into
*                  conflicts with Fortran INQUIRE and OPEN calls. Disallow output names
*                  longer than 256 characters.
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V62  *acm*  2/09 - Fix bug 1634: allow SAVE/KEEP_AXISNAMES so that child axis names
*                    arent written. They can get too long for OPeNDAP library calls
* V62  *acm*  2/09 - Fix bug 1644: Dataset title was stored in risc_buff but if mode diag
*                    is turned on, the string was overwritten. Go back to a local buffer
*                    to store the title
* V6.5  *acm* 1/10 new qualifiers for LIST netCDF4: NCFORMAT,
*                  XCHUNK,YCHUNK,ZCHUNK,TCHUNK,COMPRESSION,SHUFFLE,ENDIAN
* V6.7  *acm* 3/11 Remove warning deprecating /RIGID. There are still valid reasons
*                  to want to set the time axis to NOT be the record axis.
* V68  *acm*  1/12 ifdef double_p for double-precision ferret.
* V683 *acm*  8/12 add new SAVE/OUTTYPE=  
* V685 *acm*  6/13 new qualifier SAVE/NOBOUNDS
* V685 *acm*  6/13 Dont add SAVE/NOCOORDS
* V690 *sh*   1/14 Automated 3-argument auxiliary (curvilinear) plotting
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V710 *acm*  1/17 Ticket 1480: When writing to netCDF, check whether the region on 
*                  modulo axes is just one grid-cell longer than the axis length,e.g. 
*                  if they gave a region of /X=0:360.  If so, write one grid-cell less 
*                  data so the axis can still be a valid modulo axis.
* v720 2/2017 *acm* Ticket 2513. If /RIGID write edges but do not write bounds or 
*                  bounds attribute.
* v730 11/2017 *acm* new SAVE/ASGLOBAL to write a global attribute, see ticket 2060, issue 1332
* V740  3/2018 *acm* issue 1863: LIST/APPEND w/o /FORM=cdf to a netCDF file is an error
* vxx 4/17 *sh* list DSG variables
* V743 12/2018 *acm* issue 1907: LIST/CLOBBER/FORM=stream didn't clobber the file
* V75   5/2019 *acm* issue 1932: For err540_write_order bug, (at loop 200) make the
*                    change only for netCDF outputs not ascii multi-var listings
* V751 6/19 *acm*  DSG data cannot be subsetted by index in I,J,K,L; only by coordinates
*                  /X= /Y= /Z= /T=  or with /E= or /M=
* v751 *acm*  8/19 Issue 1946, allow for user-set string dimension length, SAVE/STRLEN
* v751 *acm*  8/19 SAVE/APPEND to dsg datasets
* v76  *acm* 11/19 New LIST/NOCOORDS to write dimensions only not coord vars. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
        include 'tmap_errors.parm'
        include 'implicit.parm'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'interp_stack.parm'
        include 'gfdl_vms.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xinterrupt.cmn'
        include 'xrisc.cmn'
	include 'xtext_info.cmn'

* calling argument declarations:

* local variable declarations
	LOGICAL		is_secure, has_dsg, do_coords
	LOGICAL		ITS_AUTO_CURVI, TM_ITSA_DSG_RAGGED,
     .			normal_file, fexist, heading, list_to_tty, do_multi,
     .			do_epic, do_cdf, do_unform, do_stream,
     .			do_comma_del, do_tab_del, do_dods, do_xml,
     .			permute, all_at_once, list_at_eof, head_enh,
     .			clobber, quiet, have_outtype, curvi_axes(nferdims), 
     .			do_shrink, as_global, arg1_quoted, re_verify

	INTEGER		TM_LENSTR1, CX_DIM_LEN, STR_CASE_BLIND_COMPARE,
     .                  STR_UPCASE, GET_MAX_STRING_LEN, DSG_WHATS_IT,
     .			TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,NF_CLOSE, 
     .			status, mr, cx, inext, ilist, wsize, vax_code,
     .			rlen, flen, reclen, i, cx1, qfile, idim,
     .			ndim, dims(nferdims), perm(7), mr_perm(max_intrp),
     .                  npermuted, recsofar, do_stream_phase, nvars,
     .                  edges_flag, keepax_flag, nn, slen, do_bounds,
     .			my_mr(max_intrp), my_cx(max_intrp), frame_bottom,
     .			nvars2list, grid, dsg_type, dset, nfeatures, var,
     .			longest_feature, nctype, cdfid, strmlen, usr_maxstrlen
	INTEGER*8	i8_size
	REAL		big_val, little_val

	CHARACTER	AUTO_FILENAME*2048, filename*2048, raw_name*2048, 
     .			buff*2048, fstatus*8, faccess*12, fformatted*12,
     .			cdf_recax*1, fposition*12, out_type*6, vname*128

* local parameter declarations
* *kob* 2/99 - add entries for quiet qualifier 
	INTEGER		slash_file, slash_append, slash_permute, slash_format,
     .			slash_seqntl, slash_rigid, slash_clobber, 
     .                  slash_quiet
	PARAMETER     (	slash_seqntl  = 15+1,
     .			slash_file    = 16+1,
     .			slash_append  = 17+1,
     .			slash_permute = 18+1,
     .			slash_format  = 19+1,
     .			slash_rigid   = 22+1,
     .			slash_clobber = 35+1,
     .                  slash_quiet   = 36+1)

* formatting information given ?
	i = qual_given( slash_format )
	IF ( i .GT. 0 ) THEN
	   CALL EQUAL_FORMAT( cmnd_buff(qual_start(i):qual_end(i)), status )
	   IF ( status .NE. ferr_ok ) GOTO 4999
	ENDIF

* did the user request to clobber previous output file versions
	clobber = qual_given( slash_clobber ) .GT. 0

* *kob* 2/99 - did the user request a quiet listing
	quiet = qual_given( slash_quiet ) .GT. 0

* special trap to create DODS cache version of NetCDF file
        IF ( list_fmt_type .EQ. plist_cdf_cache ) THEN
	   list_fmt_type = plist_default ! restore for next command
	   num_uvars_in_cmnd = 0	 ! wipe out past memory of expressions
	   CALL CDF_PREP_CACHE( cx_last, filename, nvars,
     .			   is_mr, is_cx, clobber, status )
	   IF ( status .NE. ferr_ok ) RETURN
           edges_flag = 0
           do_bounds = 0
	   do_coords = .TRUE.
	   keepax_flag = 0
	   out_type = "DFLT"
	   CALL CDF_LIST(  filename( 1:TM_LENSTR1(filename)), 
     .                    .FALSE., 'N', nvars, is_mr, is_cx,
     .			   ' ', .TRUE., pcdf_mode_cache, clobber, 
     .                    edges_flag, do_bounds, keepax_flag, out_type,  
     .                    status )
	   CALL DELETE_MRS_IN_PROGRESS		! remove temporary vars
	   RETURN
	ENDIF

* initialize
	list_at_eof  = list_append .OR. qual_given( slash_append ) .GT. 0
	do_epic	     = list_fmt_type .EQ. plist_epic
	do_cdf	     = list_fmt_type .EQ. plist_cdf
        do_unform    = list_fmt_type .EQ. plist_unformatted
        do_dods      = list_fmt_type .EQ. plist_dods
        do_stream    = list_fmt_type .EQ. plist_stream
        do_tab_del   = list_fmt_type .EQ. plist_tab_del
        do_comma_del = list_fmt_type .EQ. plist_comma_del
        do_stream_phase = 0		! for multi-variable stream
        do_xml       = list_fmt_type .EQ. plist_xml
	list_to_tty  = qual_given( slash_file ) .EQ. 0
     .            .AND. .NOT.(do_epic.OR.do_unform
     .			  .OR.do_cdf.OR.do_stream .OR. do_dods)
	normal_file  = .NOT.( list_to_tty .OR. do_epic .OR. do_cdf
     .			 .OR. do_stream .OR. do_dods) 
	permute      = qual_given( slash_permute ) .GT. 0
        npermuted    = 0
	recsofar     = 0
	
* Verboten to list to file if in secure mode
	IF (is_secure() .AND. .NOT. list_to_tty) then
	   call SPLIT_LIST(pttmode_help, err_lun,
     1	        'This command is not allowed.', 0)
	   return
	ENDIF

* in GUI mode you cannot use LIST/FORMAT without /FILE
	IF ( mode_gui.AND.list_format_given.AND.list_to_tty ) GOTO 5070

	IF ( do_stream .AND. list_at_eof ) CALL ERRMSG
     .		( ferr_not_implemented, status,
     .			'cannot /APPEND with STREAM format', *4999)

* /RIGID for netCDF ?  
* As of 2/2017, write bounds if /RIGID and irregular, rather than edges,
* unless they say /EDGES. So bounds are now treated in the default way when
* /RIGID is given.

        i = qual_given( slash_rigid )
        IF ( i .GT. 0 ) THEN
           cdf_recax = 'N'   ! no record axis
        ELSE
           cdf_recax = 'T'   ! "T" is record axis
        ENDIF

* /KEEP_AXISNAMES for netCDF ?

        i = qual_given( slash_list_keepax )
	keepax_flag = 0
        IF ( i .GT. 0 ) THEN
           keepax_flag = 1
        ENDIF

* /ASGLOBAL to write a netCDF Global Attribute? 
* Write the variable as a global attribute; conditions on this 
* variable are checked in cdf_list: 1-D, single value string or
* up to 100 values numeric.

        as_global = qual_given( slash_list_asglobal ) .GT. 0
        IF (.NOT. do_cdf .AND. as_global) CALL WARN(
     .          '/ASGLOBAL is valid only for NetCDF output. Ignored')

* /STRLEN to write string-dimension the given length.

	usr_maxstrlen = unspecified_int4
	i = qual_given(slash_list_strlen)
	IF ( i .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(i):qual_end(i)),
     .			      big_val, status )
	   usr_maxstrlen = big_val
	ENDIF	
        IF (.NOT. do_cdf .AND. usr_maxstrlen.GT.0) CALL WARN(
     .          '/STRLEN is valid only for NetCDF output. Ignored')

* /EDGES for netCDF ?
        edges_flag = 0
        IF ( qual_given(slash_list_edges) .GT. 0) edges_flag = 1

* /BOUNDS for netCDF ?
	do_bounds = 0
        IF (qual_given( slash_list_bounds ) .GT. 0) do_bounds = 1
        IF (qual_given( slash_list_nobounds ) .GT. 0) do_bounds = -1
 
* only for cdf, and not both edges and bounds.  Use BOUNDS if both specified.

        IF (do_bounds .GT. 0) THEN
           IF (.NOT. do_cdf) CALL WARN(
     .          '/BOUNDS is valid only for NetCDF output. Ignored')
           IF (edges_flag .GT. 0) THEN
             IF (qual_given( slash_list_edges ) .GT. 0)
     .            CALL WARN('Cannot have both /BOUNDS and /EDGES '//
     .           'Ignoring /EDGES')
              edges_flag = 0
           ENDIF
        ENDIF

* By default write coordinate variables, unless the user asks for /NOCOORDS

* /NOCOORDS for netCDF ?
        do_coords = .TRUE.
        IF (qual_given( slash_list_nocoords ) .GT. 0) do_coords = .FALSE.
        IF ((.NOT.do_coords) .AND. (.NOT. do_cdf)) CALL WARN(
     .         '/NOCOORDS is valid only for NetCDF output. Ignored')	

* get the requested data somehow
* (leave the variables flagged as "in-use" 7/92)
	CALL GET_PROT_CMND_DATA (  cx_last, ptype_native, status )
	IF ( status .NE. ferr_ok ) THEN      ! 8/6/92
           list_fmt_type = plist_default ! restore for next command
	   list_format_given = .FALSE.
           RETURN
        ENDIF
	do_multi     = qual_given(slash_seqntl) .EQ. 0
     .		 .AND. num_uvars_in_cmnd .GT. 1
     .		 .AND. .NOT.do_stream
c     .		 .AND. .NOT.do_tab_del
c     .		 .AND. .NOT.do_comma_del

* is this a listing of DSG variables
	has_dsg = .FALSE.
	DO ilist = 1, num_uvars_in_cmnd
	   cx = is_cx(ilist)
	   grid = cx_grid(cx)
	   IF (DSG_WHATS_IT(grid) .NE. pdsg_not_dsg) THEN 
	      dset = TM_DSG_DSET_FROM_GRID( grid )
	      has_dsg = .TRUE.
	      EXIT
	   ENDIF
	ENDDO

* Did they try to limit DSG data by index, e.g. LIST/I=  or LIST/L=

	IF (has_dsg) THEN 
	   DO idim = 1, t_dim
	      IF (cx_lo_ss(cx_cmnd, idim) .NE. unspecified_int4)   GOTO 5300
	   ENDDO
	ENDIF

	IF (do_cdf .AND. has_dsg .AND..NOT.list_at_eof) THEN 
	   buff = cmnd_buff
	   CALL ADD_DSG_VARS (dset, cmnd_buff)
	   
           CALL UNPROTECT_CMND_DATA
	   CALL PARSE_COMMAND (  cmnd_buff,
     .			     max_arg_list, max_qual_list,
     .			     len_cmnd, cmnd_num, subcmnd_num, 
     .			     num_qualifiers, qualifier_list,
     .			     qual_start, qual_end,
     .			     num_args, arg_start, arg_end,
     .			     err_lun, re_verify, arg1_quoted, status	)
	   CALL GET_PROT_CMND_DATA (  cx_last, ptype_native, status )
	   cmnd_buff = buff
	   IF ( status .NE. ferr_ok ) THEN 
              list_fmt_type = plist_default ! restore for next command
	      list_format_given = .FALSE.
              RETURN
           ENDIF

	   permute = .FALSE.

	ENDIF

	all_at_once  = do_multi.OR.do_epic.OR.do_cdf

* permutation requested ?
        DO 10 idim = 1, nferdims+1
 10     perm(idim) = idim
	IF ( permute ) THEN
* ... user-specified permutation
	   i = qual_given( slash_permute )
	   CALL EQUAL_PERMUTE( cmnd_buff(qual_start(i):qual_end(i)),
     .			       .TRUE., .FALSE., .FALSE., perm, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* consolidate the given variables and the aux vars they depend on into one list
	inext = 0
	DO i = 1, num_uvars_in_cmnd 
	   inext = inext + 1
	   my_mr(inext) = is_mr(i)
	   my_cx(inext) = is_cx(i)
* NOTE: (4/17)  THIS APPROACH TO MAKING AUXILIARY VARIABLES ACCESSIBLE
*     -- CAREFULLY PUTTING THEM ONTO THE STACK -- CAN BE REPLACED BY A MUCH
*     SIMPLER APPROACH.  PRE-LOAD AUX VARS INTO MR MEMORY AND KEEP TRACK OF
*     THEM AS PART OF THE DATASET DEFINITION.  (FOR AUXILIARY VARIABLES
*     THAT ARE THEMSELVES UVARS THIS WILL NEED SOME ADDITIONAL THOUGHT) 
* ... now put its aux vars into the list
	   IF ( ITS_AUTO_CURVI(i,curvi_axes) ) THEN
* TEMPORARY: need to ensure that 'coordinates' attributes are written to point
*	     variables to their auxiliary variables
*	     For uvars there may be no __LayerzRef_. Just uvar_aux_cat/var 
	      DO idim = 1, nferdims
	         IF (curvi_axes(idim)) THEN
	            inext = inext + 1
	            frame_bottom = is_aux_ptr(i)
	            my_mr(inext) = is_mr(frame_bottom+idim)
	            my_cx(inext) = is_cx(frame_bottom+idim)
	         ENDIF
	      ENDDO
	   ENDIF
	ENDDO
	nvars2list = inext   ! equals num_uvars_in_cmnd plus aux vars

* set up possible loop - if variable by variable output is requested
* this loop will be used - else all variables are output at once
	inext = 1
 100	mr = my_mr( inext )
	cx = my_cx( inext )

* ... determine heading request
	heading  = list_heading    ! default from SET LIST
	risc_buff = ' '
	IF ( qual_given( slash_list_nohead ) .GT. 0 ) heading = .FALSE.
* did they request an "enhanced" heading?
	i = qual_given( slash_list_head )
	IF ( i .GT. 0 ) THEN
	   heading = .TRUE.
	   CALL EQUAL_STRING(
     .                  cmnd_buff(qual_start(i):qual_end(i)),
     .                  risc_buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   head_enh = INDEX(risc_buff,"ENH") .GT. 0
	   IF ( .NOT.head_enh .AND. risc_buff.NE.' ' ) THEN
	      CALL WARN('Unrecognized option /'//
     .		         cmnd_buff(qual_start(i):qual_end(i)) )
	      CALL WARN('Valid option is /HEADING=ENHANCED')
	   ENDIF
* Enhanced heading is not consistent with KEEP_AXISNAMES flag
	   IF (keepax_flag .EQ. 1) THEN
	      CALL WARN
     .     ('/KEEP_AXISNAMES ignored. Inconsistent w/ enhanced heading.')
              keepax_flag = 0
	   ENDIF
* Enhanced heading is not consistent with /NOCOORDS flag
	   IF (.NOT. do_coords) THEN
	      CALL WARN
     .     ('/NOCOORDS ignored. Inconsistent w/ enhanced heading.')
              do_coords = .TRUE.
	   ENDIF
	      
	ELSE
	   head_enh = .FALSE.
	ENDIF
	IF ( head_enh .AND. qual_given(slash_format).EQ.0 ) GOTO 5060
	IF ( head_enh .AND. (do_comma_del.OR.do_tab_del)  ) GOTO 5060

* did they request a title?
	i = qual_given( slash_list_title )
	IF ( i .GT. 0 ) THEN
	   CALL EQUAL_STR_LC(
     .                  cmnd_buff(qual_start(i):qual_end(i)),
     .                  buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   buff = ' '
	ENDIF

* did they set an output type? allowed for netCDF and in limited way stream files

	strmlen = 0
	out_type = "DFLT"
	i = qual_given( slash_list_outtype )

	IF ( i .GT. 0 ) THEN
           IF (do_cdf .OR. do_stream .OR. do_unform) THEN
	      have_outtype = .FALSE.
	      CALL EQUAL_STRING(
     .                  cmnd_buff(qual_start(i):qual_end(i)),
     .                  out_type, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (do_cdf .OR. do_stream ) THEN 
	         have_outtype = ( INDEX(out_type,'DOU') .GT. 0 .OR.
     .                    INDEX(out_type,'FLO') .GT. 0 .OR. 
     .                    INDEX(out_type,'INT') .GT. 0 .OR.
     .                    INDEX(out_type,'SHO') .GT. 0 .OR.
     .                    INDEX(out_type,'BYT') .GT. 0 )
	         IF (.NOT. have_outtype) GOTO 5100
	      ENDIF
	      IF (do_stream .OR. do_unform .AND. have_outtype) THEN
	         strmlen = 8
		 IF (have_outtype .AND. INDEX(out_type,'FLO') .GT. 0 ) strmlen = 4
		 IF (have_outtype .AND. INDEX(out_type,'INT') .GT. 0 ) strmlen = 4
		 IF (have_outtype .AND. INDEX(out_type,'SHO') .GT. 0 ) strmlen = 2
		 IF (have_outtype .AND. INDEX(out_type,'BYT') .GT. 0 ) strmlen = 1
	      ENDIF

	   ELSE
	      CALL WARN('/OUTTYPE is valid only for NetCDF or stream output. Ignored')
           ENDIF
	ENDIF

* Get any settings for netcdf-4 output

        IF (do_cdf) THEN
	   CALL NETCDF4_LIST_SETTINGS (status)
           IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* set up the output device
	IF ( list_to_tty ) THEN
* ... output to screen
	   list_lun = ttout_lun
	ELSE
* ... output to a file
	   IF ( do_unform ) THEN
	      fformatted = 'UNFORMATTED'
	   ELSE
	      fformatted = 'FORMATTED'
	   ENDIF
* ... get full file name - from LIST/FILE= or from SET LIST/FILE=
           raw_name = ' '
           qfile = qual_given( slash_file )
           IF ( qfile .GT. 0 ) THEN
#ifdef unix
* preserve case of filename for unix
              CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  raw_name, status )
#else
* VMS filenames in upper case always
              CALL EQUAL_STRING(
     .			cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  raw_name, status )
#endif
           ENDIF
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( raw_name .EQ. ' ' .AND. do_stream_phase.EQ.0 ) THEN
* ... use default from SET LIST command
              vax_code = STR_CASE_BLIND_COMPARE(list_file(1:4),'AUTO')
	      IF ( vax_code .EQ. vms_str_success ) THEN
                 vax_code = STR_UPCASE( list_file, list_file )
	         raw_name = AUTO_FILENAME( list_file, cx, mr, rlen )
	      ELSE
	         raw_name = list_file
	      ENDIF
	   ELSEIF ( STR_CASE_BLIND_COMPARE(raw_name(1:4),'AUTO')
     .         .EQ. vms_str_success ) THEN
              vax_code = STR_UPCASE( raw_name, raw_name )
	      raw_name = AUTO_FILENAME( raw_name, cx, mr, rlen )
	   ENDIF
!	   rlen = TM_LENSTR1( raw_name )      ! ever used ???
           slen = TM_LENSTR1(raw_name)
           IF (slen .GT. 256 ) GOTO 5090
	   INQUIRE( FILE = raw_name, NAME = filename, EXIST = fexist )

	   flen = TM_LENSTR1( filename )
* ... forced heading ?
! (3/90)	   IF ( .NOT.fexist .AND. list_at_eof ) heading = .TRUE.
* ... append or not ?
	   IF ( fexist ) THEN    ! 10/93 logic change
	      IF ( list_at_eof ) THEN

* (Issue 1863 a nod to Ned who ran into this.)
* Did they ask to append to a netCDF file but without LIST/FORMAT=CDF?
* Check that a .nc or .cdf file name is in fact netcdf file. If so, its
* an error - they needed to have said SAVE or LIST/FORM=cdf.
* If its not a netCDF file type, just continue; not illegal to write
* an ascii file with extension .nc.

	         IF (.NOT. do_cdf .AND. (
     .               filename(flen-3:flen).EQ.'.cdf' .OR. 
     .               filename(flen-3:flen).EQ.'.CDF' .OR. 
     .               filename(flen-2:flen).EQ.'.nc' .OR. 
     .               filename(flen-2:flen).EQ.'.NC') ) THEN
                    
	            CALL CD_OPEN_OUT( filename, list_at_eof, cdfid, .FALSE., 
     .                     nctype, status )
	            IF (status .EQ. ferr_ok) GOTO 5110
		    status = NF_CLOSE(cdfid)
	         ENDIF

#ifdef NO_ACCESS_APPEND
	         faccess = 'SEQUENTIAL'
#else
	         faccess = 'APPEND'
		 fposition = 'APPEND'
#endif
	         fstatus = 'OLD'
	      ELSEIF (do_stream) THEN
	         IF ( do_stream_phase .EQ. 0 ) THEN

	            IF ( clobber ) THEN
	               fstatus = 'REPLACE'  ! clobber it regardless
	            ELSE
	               CALL ERRMSG
     .		   ( ferr_invalid_command,  status,
     .              'file already exists: '//raw_name,
     .              *1000 )
	            ENDIF
	         ENDIF
	      ELSEIF ( clobber ) THEN	! 11/96
	         faccess = 'SEQUENTIAL'
	         fstatus = 'REPLACE'
#ifdef F90_OPEN_FILE_CLOBBER
		 fposition = 'ASIS'
#endif
	      ELSE
	         GOTO 5050	! attempt to clobber a file
	      ENDIF
	   ELSE
	      faccess = 'SEQUENTIAL'
	      fstatus = 'NEW'
	      fposition = 'asis'
	   ENDIF
* ... open or create it
* only output listing message if user didn't specify quiet *kob* 2/99
* For dods output, file is opened in the C routine called by write_dods_fmt.

           IF (.NOT.(do_EPIC) .AND. do_stream_phase.EQ.0 
     .                    .AND. .NOT.(quiet) )
     . 	          CALL SPLIT_LIST(pttmode_bull, err_lun,
     .		  ' LISTing to file '//raw_name(1:TM_LENSTR1(raw_name))
     .								,0)

	   IF ( normal_file ) THEN
	      OPEN( UNIT	= default_list_lun,
     .		    FILE	= raw_name,
     .		    FORM	= fformatted,
#ifdef F90_OPEN_FILE_APPEND
     .		    POSITION    = fposition,
#else
     .		    ACCESS	= faccess,
#endif
     .		    STATUS	= fstatus,
     .		    ERR		= 5010 )
	   ELSEIF ( do_stream .AND. do_stream_phase.EQ.0 ) THEN	! 12/94
	      reclen = 1  ! in bytes
	      DO 60 i = 1, nvars2list
	         IF ( mr_type(my_mr(i)) .EQ. ptype_string ) THEN
	            reclen = MAX(reclen,
     .			GET_MAX_STRING_LEN(my_cx(i), my_mr(i),
     .					   memry(my_mr(i))%ptr)+1 )
	         ELSE
	            reclen = MAX(reclen,8)
	         ENDIF

 60	      CONTINUE

              IF (strmlen .GT. 0) reclen = strmlen

	      OPEN( UNIT	= default_list_lun,
     .		    FILE	= raw_name,
     .		    FORM	= 'UNFORMATTED',
     .		    ACCESS	= 'DIRECT',
     .		    STATUS	= fstatus,
#ifdef reclen_in_bytes
     .	            RECL	= reclen,
#else
     .	            RECL	= reclen/4,
#endif
     .		    ERR		= 5010 )
	      do_stream_phase = 1		! for multi-variables
	   ENDIF
	   list_lun = default_list_lun
	ENDIF

* reserve working arrays (different needs for each listing type)
	IF ( do_epic ) THEN
	   wsize = 1
	   DO 110 ilist = 1, nvars2list
 110	   wsize = MAX ( wsize, CX_DIM_LEN(t_dim,my_cx(ilist)) )
	   wsize = 3. * wsize
	ELSEIF ( do_multi ) THEN
	   wsize = nvars2list* (2*nferdims+6)
	ELSEIF ( all_at_once ) THEN
	   wsize = nvars2list
	ELSE
* ... allocate enough space for the longest axis
	   wsize = 1
	   DO 112 idim = 1, nferdims
 112	   wsize = MAX( wsize, CX_DIM_LEN(idim,cx))
	   IF (do_dods)THEN
              wsize = 1
              DO 113 idim = 1, nferdims
 113	        IF (CX_DIM_LEN(idim,cx) .GE. 1) 
     .             wsize = wsize * CX_DIM_LEN(idim,cx)
	   ENDIF
	ENDIF
	i8_size = wsize
	CALL GET_DYN_WORK_SPACE( i8_size, status )
	IF ( status .NE. ferr_ok ) GOTO 1000

* special handling of EPIC data - no permutations
	IF ( do_epic ) THEN
	   CALL EPIC_LIST( raw_name, my_mr, my_cx, 
     .			   workmem(plegacy_work_buffer)%ptr, wsize, status )
	   GOTO 1000
	ENDIF

* permute data to the desired output order

	IF ( all_at_once ) THEN
* ... if no axis order was specified determine the permutation
*      bug fix: see err540_write_order.jnl move inside loop 
* ... Ticket 1480: When writing to netCDF, test the region 
*     on modulo axes 


	   IF (.NOT.permute .AND. .NOT.do_cdf) CALL GET_CX_DIMS(cx,ndim, perm )
	   DO 200 ilist = 1, nvars2list
	      
              IF (.NOT.permute .AND.do_cdf) CALL GET_CX_DIMS(my_cx(ilist),ndim,perm)
              CALL SHRINK_FOR_MODULO (my_cx(ilist), my_mr(ilist), do_cdf,
     .				      do_shrink)
              CALL CREATE_PERMUTATION(my_cx  (ilist),
     .				      my_mr  (ilist),
     .				      perm, 
     .				      mr_perm(ilist), do_shrink, status )
              IF ( status .NE. ferr_ok ) GOTO 500
              npermuted = npermuted + 1
 200	   CONTINUE
 

* w/ the above fix, mr_perm is correct but perm needs to be reset.
	   IF (.NOT.permute) CALL GET_CX_DIMS(cx,ndim, perm )
	ELSE
* ... if no axis order was specified use the first variable as default
           IF ( .NOT.permute ) THEN
	      CALL GET_CX_DIMS( cx, ndim, perm )
* ... default listing for a line of data is a column
	      IF (ndim.EQ.1) THEN
	         i = perm(1)
	         perm(1) = perm(2)
	         perm(2) = i
	      ENDIF
	   ENDIF
	   IF (.NOT.has_dsg) THEN
	      CALL CREATE_PERMUTATION(cx,mr,perm,mr_perm,do_cdf,status)
	      IF ( status .NE. ferr_ok ) GOTO 500
              npermuted = npermuted + 1
	   ENDIF
	ENDIF

* is this a listing of DSG variables
* DSGTBD:  Is there a use case for treating DSG instance variables as 1D grids?
	has_dsg = .FALSE.
	DO ilist = 1, nvars2list
	   cx = my_cx(ilist)
	   grid = cx_grid(cx)
	   IF (DSG_WHATS_IT(grid) .NE. pdsg_not_dsg) THEN 
	      dset = TM_DSG_DSET_FROM_GRID( grid )
	      nfeatures = TM_DSG_NFEATURES( grid )
	      CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
	      longest_feature = big_val
	      has_dsg = .TRUE.
	      EXIT
	   ENDIF
	ENDDO

* DSG data and gridded data cannot be mixed in the same listing
* nor can DSG's from different datasets -- essentially separate reference 
* coordinate systems.
	IF (has_dsg) THEN
! DSGTBD Need routine CHECK_CONSISTENT_DSG ensuring vars can be listed together
! and error out if so
! Separate routine suggested if SAVE has simimlar needs
	   CONTINUE
	ENDIF

* list it
!jli: for list in xml
	IF (do_xml) THEN
*           CALL NON_ARRAY_SUBSC( mr_perm, 1 )
*           nn = mr_perm(1)
*           CALL LIST_CMND_DATA_XML( memory(1, mr_blk1(mr_perm(1))),
*     .                          nn, cx, memory(1, wkblk), 
*     .                          heading, head_enh, recsofar, perm, 
*     .                          clobber, raw_name, status)

	ELSEIF ( do_cdf ) THEN

	   IF (as_global) has_dsg = .FALSE.

	      CALL CDF_LIST(  raw_name( 1:TM_LENSTR1(raw_name)), 
     .                    list_at_eof, cdf_recax, nvars2list,
     .			  mr_perm, my_cx, buff, head_enh,
     .			  pcdf_mode_normal, clobber, edges_flag, do_bounds,
     .                    keepax_flag, out_type, quiet, as_global, 
     .                    usr_maxstrlen, do_coords, status )

* ASCII custom output for dsg-file data
	ELSEIF ( has_dsg ) THEN
! DSGTBD - make sure permutation cannot get applied??
	   CALL LIST_DSG_DATA(dset,
     .			      my_cx,
     .			      my_mr,
     .			      nvars2list,
     .			      heading,
     .			      nfeatures,
     .			      longest_feature )

	ELSEIF( all_at_once ) THEN
	   CALL LIST_MULTI(  mr_perm, my_cx, heading, perm,
     .			    wsize, workmem(plegacy_work_buffer)%ptr,status )
	ELSE
	   CALL NON_ARRAY_SUBSC( mr_perm, 1 )
           nn = mr_perm(1)
	   CALL LIST_CMND_DATA( memry(mr_perm(1))%ptr,
     .                          nn, cx,  workmem(plegacy_work_buffer)%ptr, 
     .                          heading, head_enh, recsofar, perm, 
     .                          clobber, raw_name, status)
	ENDIF

* EXIT sequence
* ... relinquish working space
 500	CALL RELEASE_DYN_WORK_SPACE

* ... close any special output device
	IF ( normal_file ) CLOSE ( UNIT = list_lun, ERR = 5010 )

* ... delete permuted data if it was created
	DO 510 ilist = 1, npermuted
	   mr = mr_perm(ilist)
	   IF ( mr_protected(mr) .EQ. mr_temporary
     .	   .OR. mr_protected(mr) .EQ. mr_table_entry_only )
     .					CALL DELETE_VARIABLE( mr )
	   IF ( .NOT.all_at_once ) THEN
* ... loop back for next variable if appropriate
	      inext = inext + 1
	      IF ( do_stream) do_stream_phase = do_stream_phase + 1
	      IF ( status.EQ.ferr_OK .AND. inext.LE.nvars2list ) THEN
	         GOTO 100
	      ELSE
	         GOTO 1000
	      ENDIF
	   ENDIF
 510	CONTINUE

 1000   list_fmt_type = plist_default ! restore for next command
	list_format_given = .FALSE.
        CALL UNPROTECT_CMND_DATA
	IF ( do_stream .AND. do_stream_phase.GT.0)
     .			CLOSE ( UNIT = list_lun, ERR = 4999 )
 4999   RETURN

* error exit(s)
 5000   GOTO 1000
 5010	CALL ERRMSG( ferr_erreq,status,filename(:flen),*1000 )
 5040	CALL ERRMSG( ferr_invalid_command, status,
     .			'/EXTEND only allowed with CDF format', *1000 )
 5050	CALL ERRMSG( ferr_invalid_command, status,
     .			'/CLOBBER not specified and file exists: '
     .			//raw_name, *1000 )
 5060	CALL ERRMSG( ferr_invalid_command, status,
     .		'LIST/HEADING=enhanced needs a /FORMAT qualifier'//pCR//
     .		'	(format cannot be tab or comma)', *1000 )
 5070	CALL ERRMSG( ferr_invalid_command, status,
     .		'In GUI mode LIST/FORMAT reguires a named /FILE=', *1000 )
 5080	CALL ERRMSG( ferr_invalid_command, status,
     .			'/ORDER not allowed with V permutation (yet)', *1000 )
     
 5090	CALL ERRMSG( ferr_prog_limit, status,
     .  'Cannot write to file w/ name longer than 256 characters: '//pCR//
     .  'Limitation of Fortran INQUIRE and OPEN calls '	
     .			, *5000 )
     
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'Unrecognized option/'//
     .          cmnd_buff(qual_start(i):qual_end(i)) //pCR//
     .		' Valid options are /OUTTYPE=DOUBLE,FLOAT,INT,SHORT,or BYTE',
     .		 *1000 )

 5110	CALL ERRMSG( ferr_invalid_command, status,
     .			'/LIST/APPEND to netCDF file requires /FORMAT=cdf', *1000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .   '/LIST/APPEND to netCDF ragged-array DSG file not implemented', *1000 )
 5300	CALL ERRMSG( ferr_syntax, status, 
     .	'Limits by subscript /'//ss_dim_name(idim)//
     .	'= invalid for DSG data. Use /'//ww_dim_name(idim)//
     .	'= limits or select by feature', *1000 )

	END


