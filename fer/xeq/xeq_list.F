	SUBROUTINE XEQ_LIST( memory )

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
*				 AIX fortran.  Add ifdef NO_OPEN_RECORDTYPE.
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
*		      "ifndef sgi" to prevent reclen_in_bytes being set
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

#ifdef unix
        include 'tmap_errors.parm'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
        include 'gfdl_vms.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xinterrupt.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
        INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations
	LOGICAL		normal_file, fexist, heading, list_to_tty, do_multi,
     .			do_epic, do_gt, do_cdf, do_unform, do_stream,
     .			do_comma_del, do_tab_del,
     .			permute, all_at_once, list_at_eof, head_enh,
     .			clobber, quiet
	INTEGER		TM_LENSTR1, CX_DIM_LEN, STR_CASE_BLIND_COMPARE,
     .                  STR_UPCASE, GET_MAX_STRING_LEN,
     .			status, mr, cx, inext, ilist, wsize, vax_code,
     .			rlen, flen, reclen, i, cx1, qfile, idim,
     .			ndim, wkblk, perm(5), mr_perm(max_intrp),
     .                  npermuted, recsofar, do_stream_phase, nvars
	CHARACTER	AUTO_FILENAME*200, filename*200, raw_name*200,
     .			buff*80, fstatus*8, faccess*12, fformatted*12,
     .			cdf_recax*1,
     .			fposition*12

* local parameter declarations
* *kob* 2/99 - add entries for quiet qualifier 
	INTEGER		slash_file, slash_append, slash_permute, slash_format,
     .			slash_seqntl, slash_rigid, slash_clobber, slash_quiet
	PARAMETER     (	slash_seqntl  = 11+1,
     .			slash_file    = 12+1,
     .			slash_append  = 13+1,
     .			slash_permute = 14+1,
     .			slash_format  = 15+1,
     .			slash_rigid   = 18+1,
     .			slash_clobber = 27+1,
     .                  slash_quiet   = 28+1)

* one lonely function declaration
	LOGICAL is_secure

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
     .			  memory, is_mr, is_cx, clobber, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL CDF_LIST( memory, filename( 1:TM_LENSTR1(filename)), 
     .                    .FALSE., 'N', nvars, is_mr, is_cx,
     .			   ' ', .TRUE., pcdf_mode_cache, clobber, status )
	   CALL DELETE_MRS_IN_PROGRESS		! remove temporary vars
	   RETURN
	ENDIF

* initialize
	list_at_eof  = list_append .OR. qual_given( slash_append ) .GT. 0
	do_epic	     = list_fmt_type .EQ. plist_epic
	do_gt	     = list_fmt_type .EQ. plist_gt
	do_cdf	     = list_fmt_type .EQ. plist_cdf
        do_unform    = list_fmt_type .EQ. plist_unformatted
        do_stream    = list_fmt_type .EQ. plist_stream
        do_tab_del   = list_fmt_type .EQ. plist_tab_del
        do_comma_del = list_fmt_type .EQ. plist_comma_del
        do_stream_phase = 0		! for multi-variable stream
	list_to_tty  = qual_given( slash_file ) .EQ. 0
     .            .AND. .NOT.(do_epic.OR.do_gt.OR.do_unform
     .			  .OR.do_cdf.OR.do_stream)
	normal_file  = .NOT.( list_to_tty .OR. do_epic .OR. do_cdf
     .			 .OR. do_stream ) 
	permute      = qual_given( slash_permute ) .GT. 0
        npermuted    = 0
	recsofar     = 0
	
* Verboten to list to file if in secure mode
	IF (is_secure() .AND. .NOT. list_to_tty) then
	   call SPLIT_LIST(pttmode_help, ttout_lun,
     1	        'This command is not allowed.', 0)
	   return
	ENDIF

* in GUI mode you cannot use LIST/FORMAT without /FILE
	IF ( mode_gui.AND.list_format_given.AND.list_to_tty ) GOTO 5070

	IF ( do_stream .AND. list_at_eof ) CALL ERRMSG
     .		( ferr_not_implemented, status,
     .			'cannot /APPEND with STREAM format', *4999)

#ifdef AIX_XLF
	IF ( do_gt ) CALL ERRMSG( ferr_not_implemented, status,
     .			'GT format not available on AIX systems', *4999)
#endif

* /RIGID for netCDF ?
        i = qual_given( slash_rigid )
        IF ( i .GT. 0 ) THEN
           cdf_recax = 'N'   ! no record axis
        ELSE
           cdf_recax = 'T'   ! "T" is record axis
        ENDIF

* get the requested data somehow
* (leave the variables flagged as "in-use" 7/92)
	CALL GET_PROT_CMND_DATA ( memory, cx_last, ptype_native, status )
	IF ( status .NE. ferr_ok ) THEN      ! 8/6/92
           list_fmt_type = plist_default ! restore for next command
	   list_format_given = .FALSE.
           RETURN
        ENDIF
	do_multi     = qual_given(slash_seqntl) .EQ. 0
     .		 .AND. num_uvars_in_cmnd .GT. 1
     .		 .AND. .NOT.do_stream
     .		 .AND. .NOT.do_tab_del
     .		 .AND. .NOT.do_comma_del

	all_at_once  = do_multi.OR.do_epic.OR.do_gt.OR.do_cdf

* permutation requested ?
        DO 10 idim = 1, 5 ! 3/92 SUN port addition
 10     perm(idim) = idim
	IF ( permute ) THEN
* ... user-specified permutation
	   IF ( do_gt ) GOTO 5030
	   i = qual_given( slash_permute )
	   CALL EQUAL_PERMUTE( cmnd_buff(qual_start(i):qual_end(i)),
     .			       .TRUE., .FALSE., .FALSE., perm, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* set up possible loop - if variable by variable output is requested
* this loop will be used - else all variables are output at once
	inext = 1
 100	mr = is_mr( inext )
	cx = is_cx( inext )

* ... determine heading request
	heading  = list_heading    ! default from SET LIST
	buff = ' '
	IF ( qual_given( slash_list_nohead ) .GT. 0 ) heading = .FALSE.
* did they request an "enhanced" heading?
	i = qual_given( slash_list_head )
	IF ( i .GT. 0 ) THEN
	   heading = .TRUE.
	   CALL EQUAL_STRING(
     .                  cmnd_buff(qual_start(i):qual_end(i)),
     .                  buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   head_enh = INDEX(buff,"ENH") .GT. 0
	   IF ( .NOT.head_enh .AND. buff.NE.' ' ) THEN
	      CALL WARN('Unrecognized option /'//
     .		         cmnd_buff(qual_start(i):qual_end(i)) )
	      CALL WARN('Valid option is /HEADING=ENHANCED')
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
	   INQUIRE( FILE = raw_name, NAME = filename, EXIST = fexist )
	   flen = TM_LENSTR1( filename )
* ... forced heading ?
! (3/90)	   IF ( .NOT.fexist .AND. list_at_eof ) heading = .TRUE.
* ... append or not ?
	   IF ( fexist ) THEN    ! 10/93 logic change
	      IF ( list_at_eof ) THEN
#ifdef unix
                 IF ( do_gt ) CALL ERRMSG( ferr_invalid_command,
     .              status,
     .              'appending to GT format not implemented on Unix',
     .              *1000 )
#endif
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
	               fstatus = 'OLD'
	            ELSE
	               CALL ERRMSG
     .		   ( ferr_invalid_command,  status,
     .              'file already exists: '//raw_name,
     .              *1000 )
	            ENDIF
	         ENDIF
	      ELSEIF ( clobber ) THEN	! 11/96
	         faccess = 'SEQUENTIAL'
	         fstatus = 'OLD'
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
	   IF (.NOT.(do_EPIC) .AND. do_stream_phase.EQ.0 
     .                    .AND. .NOT.(quiet) )
     . 	          CALL SPLIT_LIST(pttmode_bull, ttout_lun,
     .		  ' LISTing to file '//raw_name(1:TM_LENSTR1(raw_name))
     .								,0)

	   IF ( do_gt ) THEN
	      reclen = 8	! minimum
	      DO 50 i = 1, num_uvars_in_cmnd
	         cx1 = is_cx( i )
	         reclen = MAX( reclen, 2+CX_DIM_LEN(x_dim,cx1)  )
 50	      CONTINUE
* modified below ifdef logic.  reclen_in_bytes was only getting set, not
* checked as it should have been for other systems  *kob* 5/97
c *kob* -7/96 reclen_in_bytes not appropriate for sgi
#ifdef sun
# ifndef sgi
#    define reclen_in_bytes
# endif
#endif
#ifdef reclen_in_bytes
              reclen = reclen * 4    ! SUN record length in bytes
#endif
#ifdef unix
	      OPEN( UNIT	= default_list_lun,
     .		    FILE	= raw_name,
     .		    RECL	= reclen,
#ifdef NO_OPEN_RECORDTYPE
#else
     .		    RECORDTYPE	= 'FIXED',
#endif
     .		    FORM	= 'UNFORMATTED',
     .		    ACCESS	= 'DIRECT',
     .		    STATUS	= fstatus,
     .		    ERR		= 5010 )
	   ELSEIF ( normal_file ) THEN
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
	      DO 60 i = 1, num_uvars_in_cmnd
	         IF ( mr_type(is_mr(i)) .EQ. ptype_string ) THEN
	            reclen = MAX(reclen,
     .			GET_MAX_STRING_LEN(is_cx(i), is_mr(i),
     .					   memory(1,mr_blk1(is_mr(i))))+1 )
	         ELSE
	            reclen = MAX(reclen,4)
	         ENDIF
 60	      CONTINUE
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
#else
! VMS
	      OPEN( UNIT	= default_list_lun,
     .		    DEFAULTFILE	= 'FERRET.DAT',
     .		    FILE	= raw_name,
     .		    RECL	= reclen,
     .		    RECORDTYPE	= 'FIXED',
     .		    FORM	= 'UNFORMATTED',
     .		    ACCESS	= faccess,
     .		    STATUS	= fstatus,
     .		    ERR		= 5010 )
	   ELSEIF ( normal_file ) THEN
	      OPEN( UNIT	= default_list_lun,
     .		    DEFAULTFILE	= 'FERRET.DAT',
     .		    FILE	= raw_name,
     .		    FORM	= fformatted,
     .		    ACCESS	= faccess,
     .		    STATUS	= fstatus,
     .		    ERR		= 5010 )
	   ELSEIF ( do_stream ) THEN			! 12/94
	      OPEN( UNIT	= default_list_lun,
     .		    FILE	= raw_name,
     .		    FORM	= 'UNFORMATTED',
     .		    ACCESS	= 'DIRECT',
     .		    STATUS	= 'NEW',
     .	            RECL	= 1,
     .		    ERR		= 5010 )
	      do_stream_phase = 1		! for multi-variables
#endif
	   ENDIF
	   list_lun = default_list_lun
	ENDIF

* reserve working arrays (different needs for each listing type)
	IF ( do_epic ) THEN
	   wsize = 1
	   DO 110 ilist = 1, num_uvars_in_cmnd
 110	   wsize = MAX ( wsize, CX_DIM_LEN(t_dim,is_cx(ilist)) )
	   wsize = 3. * wsize
	ELSEIF ( do_multi ) THEN
	   wsize = num_uvars_in_cmnd*14
	ELSEIF ( all_at_once ) THEN
	   wsize = num_uvars_in_cmnd
	ELSE
* ... allocate enough space for the longest axis
	   wsize = 1
	   DO 112 idim = 1, 4
 112	   wsize = MAX( wsize, CX_DIM_LEN(idim,cx))
	ENDIF
	CALL GET_WORK_SPC( wsize, wkblk, status )
	IF ( status .NE. ferr_ok ) GOTO 1000

* special handling of EPIC data - no permutations
	IF ( do_epic ) THEN
	   CALL EPIC_LIST( raw_name, is_mr, is_cx, 
     .			   memory(1, wkblk), wsize, status )
	   GOTO 1000
	ENDIF

* permute data to the desired output order
	IF ( all_at_once ) THEN
* ... if no axis order was specified determine the permutation
	   IF (.NOT.permute) CALL GET_CX_DIMS(cx,ndim, perm )
	   DO 200 ilist = 1, num_uvars_in_cmnd
	      CALL CREATE_PERMUTATION(	memory,
     .					is_cx  (ilist),
     .					is_mr  (ilist),
     .					perm,
     .					mr_perm(ilist), status )
	      IF ( status .NE. ferr_ok ) GOTO 500
              npermuted = npermuted + 1
 200	   CONTINUE
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
	   CALL CREATE_PERMUTATION(memory,cx,mr,perm,mr_perm,status)
	   IF ( status .NE. ferr_ok ) GOTO 500
           npermuted = npermuted + 1
	ENDIF

* list it
	IF ( do_gt ) THEN
	   CALL TMAP_LIST( memory, list_lun, reclen, num_uvars_in_cmnd,
     .			   is_mr, is_cx, memory(1, wkblk), status )
	ELSEIF ( do_cdf ) THEN
	   CALL CDF_LIST( memory, raw_name( 1:TM_LENSTR1(raw_name)), 
     .                    list_at_eof, cdf_recax, num_uvars_in_cmnd,
     .			  mr_perm, is_cx, buff, head_enh,
     .			  pcdf_mode_normal, clobber, status )
	ELSEIF( all_at_once ) THEN
	   CALL LIST_MULTI( memory, mr_perm, is_cx, heading, perm,
     .			    wsize, memory(1, wkblk), clobber, status )
	ELSE
	   CALL NON_ARRAY_SUBSC( mr_perm, 1 )
	   CALL LIST_CMND_DATA( memory(1, mr_blk1(mr_perm(1))), mr_perm,
     .				cx, memory(1, wkblk), heading, head_enh,
     .				recsofar, perm, clobber, status )
	ENDIF

* EXIT sequence
* ... relinquish working space
 500	CALL RELEASE_WORK_SPC

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
	      IF ( status.EQ.ferr_OK .AND. inext.LE.num_uvars_in_cmnd ) THEN
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
 5030	CALL ERRMSG( ferr_invalid_command, status,
     .			'/ORDER not allowed with GT output (yet)', *1000 )
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
	END


