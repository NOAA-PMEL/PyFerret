	SUBROUTINE TM_READ_GRIDFILE ( gfile_name, status)
* Reads in grid and line definitions and coordinates from a specified .GRD file
*
* revision 0.00 - 05/07/87 - by Mark Verschell for PMEL/TMAP
* revision 1.00 - 05/17/87 - by Mark Verschell for PMEL/TMAP
* revision 1.10 - 08/17/88 - added removal of grid/lines defined by bad file
* revision 1.11 - 08/18/88 - loops from max,1 to find empty storage
* revision 1.12 - 08/20/88 - old name GT_READ_GRIDFILE, called from TM_INIT_DSET
* revision 2.00 - 10/05/88 - new procedures for detection of duplicates
*			     Duplicated name/defn is ignored.
*			     Duplicated name, new defn is an error.
* revision 2.10 - 10/25/88 - Modulo flag added for lines definitions
* revision 2.11 - 11/21/88 - variable used checks changed to char_initXX
* revision 2.20 - 11/22/88 - Removed references to dset_num.
* revision 2.30 - 11/29/88 - True check for unknown and normal axis. Grid_line
*			     now flagged with mpsunkn or mpsnorm.
* revision 2.40 - 12/12/88 - Changes to make use of next_line_mem_pos, including
*			     resetting it if error occurs, and incrementing it
*			     only when line defined and checked for duplication.
* revision 2.41 - 12/13/88 - Added call TM_FIND_*_SLOT to determine open space
* revision 2.50 - 12/15/88 - Reads E/W as long. N/S as lat. for regular axis
* revision 2.60 - 12/19/88 - Reads time axis info, added TIME definition to the
*			     gridfile format to get t0 and tunit for timeaxis
* revision 2.70 - 01/13/89 - Any line that isn't a GRID/LINE/TIME/COORD/Comment 
*			     is not a warning message; doesn't read tunit from
*			     TIME line - gets it from un_convert
* revision 2.71 - 01/23/89 - Moved next_line_mem_pos save to before INQUIRE.
* revision 2.72 - 02/17/89 - Fixed next_line_mem_pos error (1 less than correct)
* revision 2.73 - 01/20/91 - added ifdef unix for default file
* Unix/RISC port - 4/24/91 *sh*: uses TM_INQ_PATH to locate grid file
*                              : changed "WARNING" to "Note"
*                              : allowed axis units of "none" or "n/a"
*                        10/91 : "grd" --> ".grd" for default extension
* bug fix - 10/2/91 - commented out statments 9000 and 9710 because there were
*                      no paths to these statements.  This was caught by ver.3
*                      of f77, but not ver. 2.1  ---kob
* bug fix - 6/24/92 - allow blank lines in gridfile
* bug fix - 1/4/93 *sh* - many error exits closed the unopened lunit
*                       - I added variable "opened"
* IBM Port- 4apr94 *kob* : readonly not allowed in file open statement
*			 : with IBM xlf compiler, added "ifdef" check
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* *sh* 9/95 (for V420): axes need not have T0 to be valid.  Disable this check.
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* arguments
	CHARACTER*(*)	gfile_name
	INTEGER*4	status

#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xio.cmn_text'
	external xio_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
        include 'tmap_format/xsys_names.cmn_text'
        external xsys_names_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
        INCLUDE 'TMAP_FORMAT:XSYS_NAMES.CMN'
#endif

* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	CHARACTER*6	grid_define, line_define, coords_define,
     .			time_define, comment_char
        CHARACTER       def_ext*4
	CHARACTER*13	TM_STRING
	CHARACTER*25	buff(11), newline
	CHARACTER*80	def_error
        CHARACTER*128   TM_INQ_PATH, gfile_path
	CHARACTER*132	TM_CMPRSS, input_buf, cmprss_buf
	INTEGER*4	lunit, i, j, ipt1, ipte, npts, gdup, lcnt,
     .			cline, tline, temp
	PARAMETER	(temp = 0)		!kob 10/96
	INTEGER*4	p_gname, p_gline(4), p_grot, p_lname, p_llen,
     .			p_ldirect, ngrid, nline, istat, p_goutp(4),
     .			p_lunits, p_lstart, p_ldelta, p_cform, p_lstore,
     .			p_modulo, p_lbox0, p_t0, inner, outer, newgrid,
     .			g_used, l_used, orig_line_mem_pos, dim
	LOGICAL		reg_coords, irreg_coords
	REAL*8		rdummy
	INTEGER*4	TM_UNIT_ID, STR_CASE_BLIND_COMPARE,
     .			TM_FIND_GRID_SLOT, TM_FIND_LINE_SLOT
	LOGICAL		file_exist, opened
	LOGICAL		TM_HAS_STRING, TM_GRID_MATCH, TM_LINE_MATCH,
     .			TM_CHECK_BOX_LOC

	DATA		p_gline/4, 6, 8, 10/
	DATA		p_goutp/5, 7, 9, 11/
	DATA            def_ext/'.grd'/

	PARAMETER	(grid_define	= 'GRID  ',
     .			 line_define	= 'LINE  ',
     .			 coords_define	= 'COORDS',
     .			 time_define	= 'TIME  ',
     .			 comment_char	= '*     ')

	PARAMETER	(p_gname	= 2,
     .			 p_grot		= 3,
     .			 p_lname	= 2,
     .			 p_llen		= 3,
     .			 p_ldirect	= 4,
     .			 p_lunits	= 5,
     .			 p_lstore	= 6,
     .			 p_lstart	= 7,
     .			 p_ldelta	= 8,
     .			 p_modulo	= 9,
     .			 p_cform	= 3,
     .			 p_lbox0	= 4,
     .			 p_t0		= 3)

* Store original start of free line memory
	orig_line_mem_pos = next_line_mem_pos

* Find full file specification
        opened = .FALSE.       ! 1/93
        gfile_path = TM_INQ_PATH( gfile_name, tmap_grids, def_ext,
     .                             .TRUE., status )
        IF ( status .NE. merr_ok ) GOTO 9999

* Check to see if this gridfile has already been read
  	DO 100 i = 1, maxgridfiles
	  istat = STR_CASE_BLIND_COMPARE(grid_filename(i),gfile_path)
	  IF (istat .EQ. str_match) THEN
	    status = merr_no_action
	    RETURN
	  ENDIF
  100	CONTINUE

* Find locations to store info from this gridfile
	DO 110 i = maxgridfiles,1,-1
	  IF (grid_filename(i) .NE. char_init128) THEN
	    IF (i .EQ. maxgridfiles) GOTO 9050     !Too many grid files to store
	    newgrid = i + 1
	    GOTO 120
	  ENDIF
  110	CONTINUE
	newgrid = 1
  120	grid_filename(newgrid) = gfile_path

* How many grid spaces used 
	status = TM_FIND_GRID_SLOT ( ngrid )
	IF (status .NE. merr_ok) GOTO 9200
	ngrid = ngrid - 1
	g_used = ngrid

* How many line spaces used 
	status = TM_FIND_LINE_SLOT ( nline )
	IF (status .NE. merr_ok) GOTO 9500
	nline = nline - 1
	l_used = nline

* get a free logical unit #
	CALL LIB_GET_LUN (lunit)

* open grid filename
	OPEN 	(UNIT = lunit,
     .		 FILE = grid_filename(newgrid),
#ifdef READONLY_ALLOWED
     .		 READONLY,
#endif
     .		 STATUS = 'OLD',
     .		 ERR = 9010)
        opened = .TRUE.

* read in record from grid file
  170	READ	(lunit, '(A)',END=330,ERR=9010) input_buf
* remove extraneous spaces and tabs
	cmprss_buf = TM_CMPRSS(input_buf)
* blank buff string
	DO 180 i =1,11
	  buff(i) = ' '
  180	CONTINUE

********************************************************************************
*                                 GRID DEFINITION                              *
********************************************************************************
* Check if grid definition, and if so seperate necessary elements
	IF (cmprss_buf(1:4) .EQ. grid_define) THEN
	  CALL TM_BREAK_STR(cmprss_buf, buff, 11)

* store info in common - temporary location 0
	  READ  (buff(p_grot), *, ERR=9600) grid_rotation(temp)

* check to see if line defined - if so: place in common - if not: error
	  DO 210 i = 1,4
	    newline = buff(p_gline(i))
	    IF (STR_CASE_BLIND_COMPARE(newline(:7),'UNKNOWN') .EQ.
     .		  str_match) THEN
	      grid_line(i,temp) = mpsunkn
	      GOTO 210
	    ENDIF
	    IF (STR_CASE_BLIND_COMPARE(newline(:6),'NORMAL')  .EQ.
     .		  str_match) THEN
	      grid_line(i,temp) = mpsnorm
	      GOTO 210
	    ENDIF
	    DO 190 j = 1, nline
	      istat = STR_CASE_BLIND_COMPARE ( line_name(j), newline )
	      IF (istat .EQ. str_match) THEN
	        grid_line(i,temp) = j
		GOTO 200
	      ENDIF
  190	    CONTINUE
	    GOTO 9300  !Doesn't exist

*  Inner or outer product?
  200	    outer = STR_CASE_BLIND_COMPARE('OUTER',buff(p_goutp(i)))
	    inner = STR_CASE_BLIND_COMPARE('INNER',buff(p_goutp(i)))
	    IF (inner .EQ. str_match) THEN
	      grid_out_prod(i,temp) = .FALSE.
	    ELSE IF (outer .EQ. str_match) THEN
	      grid_out_prod(i,temp) = .TRUE.
	    ELSE
	      GOTO 9610
	    ENDIF
  210	  CONTINUE

* Check for unique grid name
	  DO 220 i = 1, ngrid
	    istat = STR_CASE_BLIND_COMPARE(grid_name(i),buff(p_gname))
	    IF (istat .EQ. str_match) THEN
	      IF (TM_GRID_MATCH(i,temp)) THEN
	        GOTO 170		! Already defined - no need to re-store
	      ELSE
	        GOTO 9100		! Whoops - same name - different def.
	      ENDIF
	    ENDIF
  220	  CONTINUE

* Name is unique .. Next location to store info; if too many grids - give error
	  status = TM_FIND_GRID_SLOT ( ngrid )
	  IF (status .NE. merr_ok) GOTO 9200

* Seems ok - transfer to permanent spot
	  CALL TM_COPY_GRID ( temp, ngrid )
* Store grid name as flag of completed grid definition
	  grid_name(ngrid) = buff(p_gname)

********************************************************************************
*                                 LINE DEFINITION                              *
********************************************************************************
* Check if line definition, and if so seperate necessary elements
	ELSE IF (cmprss_buf(1:4) .EQ. line_define) THEN
	  CALL TM_BREAK_STR(cmprss_buf, buff, 9)

* store info in common - temporary location 0
	  READ  (buff(p_llen), *, ERR=9700) line_dim(temp)
	  line_direction(temp) = buff(p_ldirect)
	  line_units(temp) = buff(p_lunits)
	  line_unit_code(temp) = TM_UNIT_ID(buff(p_lunits))
	  line_subsc1(temp) = unspecified_int4  !coordinates given later?
	  reg_coords = TM_HAS_STRING( buff(p_lstore), 'START')
	  irreg_coords = TM_HAS_STRING( buff(p_lstore), 'GIVEN')
	  line_modulo(temp) = TM_HAS_STRING( buff(p_modulo), 'MODULO')
	  line_t0(temp) = char_init20
	  line_tunit(temp) = real4_init
	  IF (reg_coords) THEN
	    IF (line_direction(temp) .EQ. 'WE') dim = x_dim
	    IF (line_direction(temp) .EQ. 'SN') dim = y_dim
	    IF (line_direction(temp) .EQ. 'UD') dim = z_dim
	    IF (line_direction(temp) .EQ. 'TI') dim = t_dim
	    CALL TM_TRANSLATE_AXIS(buff(p_lstart),dim,line_start(temp),
     .	                           status)
	    IF (status .NE. merr_ok) GOTO 9950		!error reported already
	    READ  (buff(p_ldelta), *, ERR=9720) line_delta(temp)
	    IF (line_delta(temp) .LE. 0.0D0) GOTO 9730
	    line_regular(temp) = .TRUE.
	  ELSE
	    line_delta(temp) = unspecified_val8
	    line_regular(temp) = .FALSE.
	  ENDIF

* Check for unknown units or orientation
	  IF (line_unit_code(temp) .EQ. 0) THEN
	     istat = STR_CASE_BLIND_COMPARE(line_units(temp),'none')
	     IF (istat .EQ. str_match) THEN
                line_units(temp) = ' '
             ELSE
	        input_buf = ' '
                WRITE (input_buf, *) 'cannot convert axis units: ',
     .				  line_units(temp)
	        CALL TM_NOTE( input_buf, tooplu)
             ENDIF
          ENDIF
	  DO 240 i = 1, max_axis_orients
	    IF (line_direction (temp) .EQ. axis_orients( i ) ) GOTO 250
  240	  CONTINUE
	  input_buf = ' '
	  WRITE (input_buf, *) 'UNKNOWN LINE ORIENTATION : ',
     .	                    line_direction(temp),' on ',buff(p_lname)
	  CALL TM_NOTE( input_buf, tooplu)

* If line is irregular - store it now and check it later
  250	  IF (reg_coords) THEN
* Check for unique line name
	    DO 260 i  = 1, nline
	      istat = STR_CASE_BLIND_COMPARE(line_name(i),buff(p_lname))
	      IF (istat .EQ. str_match) THEN
* Check for unique definition
	        IF (TM_LINE_MATCH(i,temp)) THEN
	          GOTO 170		! Already defined - no need to re-store
	        ELSE
	          GOTO 9400		! Whoops - same name - different def.
	        ENDIF
	      ENDIF
  260	    CONTINUE
	  ENDIF
* Name is unique .. Next location to store info; if too many lines - give error
	  status = TM_FIND_LINE_SLOT ( nline )
	  IF (status .NE. merr_ok) GOTO 9500

* Seems ok - transfer to permanent spot
	  CALL TM_COPY_LINE ( temp, nline )
* Store line name as a flag of completed line definition
	  line_name(nline) = buff(p_lname)

********************************************************************************
*                                COORD DEFINITION                              *
********************************************************************************
* Check if coord definition, and if so seperate necessary elements
	ELSE IF (cmprss_buf(1:6) .EQ. coords_define) THEN
	  CALL TM_BREAK_STR(cmprss_buf, buff, 4)

* Locate the line this defines - already defined must be dupl. name or error
	  DO 270 i = l_used+1, nline
	    IF (line_delta(i) .EQ. unspecified_val8) THEN
	      istat = STR_CASE_BLIND_COMPARE(line_name(i),buff(p_lname))
	      IF (istat .EQ. str_match) THEN
* Make sure it hasn't been already defined
	        IF (line_subsc1(i) .EQ. unspecified_int4) THEN
	          cline = i
	          GOTO 280
	        ENDIF
	      ENDIF
	    ENDIF
  270	  CONTINUE
	  GOTO 9310 !No match found - error

* Allocate some storage for the coordinates
  280	  line_subsc1(cline) = next_line_mem_pos
	    IF ((line_subsc1(cline)+2*line_dim(cline)) .GT.
     .	        maxlinestore) GOTO 9800
* Store initial box lower bound (start of axis)
	  ipt1 = line_subsc1(cline)
	  npts = line_dim(cline)
	  ipte = ipt1 + npts-1
	  READ (buff(p_lbox0), *, ERR=9740) line_mem(ipte+1)

* Read line coords and box sizes from file
	  READ  (lunit, buff(p_cform),ERR=9750)
     .	   (line_mem(i),line_mem(i+npts+1),i=ipt1,ipte)

* Check for invalid ordering of data or box sizes
	  DO 290 i=ipt1+1,ipte
	    IF (line_mem(i) .LE. line_mem(i-1)) GOTO 9760
  290	  CONTINUE
	  DO 300 i=ipt1,ipte
	    IF (line_mem(i+npts+1) .LT. 0.0D0 ) GOTO 9770
  300	  CONTINUE

* Change box sizes to box locations
	  DO 310 i=ipt1,ipte
	    line_mem(i+npts+1) = line_mem(i+npts) + line_mem(i+npts+1)
  310	  CONTINUE

* Check to see that each data point is contained inside it's box
	  IF (.NOT. TM_CHECK_BOX_LOC(cline)) GOTO 9780

* Check for unique name
	  DO 320 i  = 1, nline
	    IF (i .EQ. cline) GOTO 320
	    istat = STR_CASE_BLIND_COMPARE(line_name(i),line_name(cline))
	    IF (istat .EQ. str_match) THEN
* Check for unique definition
	      IF (TM_LINE_MATCH(i,cline)) THEN
	        line_name(cline) = ' '	! Already defined - no need to re-store
	        GOTO 170
	      ELSE
	        GOTO 9400		! Whoops - same name - different def.
	      ENDIF
	    ENDIF
  320	  CONTINUE

* Store next line mem position - good line definition
	  next_line_mem_pos = next_line_mem_pos + 2*line_dim(cline) + 1

********************************************************************************
*                             TIME LINE DEFINITION                             *
********************************************************************************
* Check if time line definition, and if so seperate necessary elements
	ELSE IF (cmprss_buf(1:4) .EQ. time_define) THEN
	  CALL TM_BREAK_STR(cmprss_buf, buff, 4)

* Locate the line this defines - already defined must be dupl. name or error
	  DO 370 i = l_used+1, nline
	    IF (line_direction(i) .EQ. 'TI') THEN
	      istat = STR_CASE_BLIND_COMPARE(line_name(i),buff(p_lname))
	      IF (istat .EQ. str_match) THEN
* Make sure it hasn't been already defined
	        IF (line_t0(i) .EQ. char_init20) THEN
	          tline = i
	          GOTO 380
	        ENDIF
	      ENDIF
	    ENDIF
  370	  CONTINUE
	  GOTO 9320 !No match found - error

* Found available slot - let's store values
  380	  CALL TM_TRANSLATE_AXIS(buff(p_t0),t_dim,rdummy,status)
	  IF (status .NE. merr_ok) GOTO 9910
	  line_t0(tline) = buff(p_t0)
	  line_tunit(tline) = un_convert(line_unit_code(tline))

********************************************************************************
*                                 COMMENT LINE                                 *
********************************************************************************
	ELSE IF (cmprss_buf(1:1) .EQ. ' ' ) THEN   ! allow blank lines 6/92
           GOTO 170            
	ELSE IF (cmprss_buf(1:1) .NE. comment_char) THEN
	  input_buf = ' '
	  WRITE (input_buf,*) 'Unrecognized line in gridfile',
     .	                   grid_filename(newgrid)
	  CALL TM_NOTE( input_buf, tooplu)
	  CALL TM_NOTE( cmprss_buf(1:79), tooplu)
	ENDIF
	GOTO 170

********************************************************************************
*                                FINAL CLEAN-UP                                *
********************************************************************************
  330	DO 340 i = 1, nline
	  IF  ( line_name(i) .NE. char_init16 ) THEN
* Are there lines left without coordinates?
	    IF (line_subsc1(i) .EQ. unspecified_int4 .AND.
     .	        line_delta(i) .EQ. unspecified_val8 ) GOTO 9900
* Are there time axis left without time info?
! disabled T0 check 9/95
!	    IF (line_direction(i) .EQ. 'TI' .AND.
!     .	        line_t0(i) .EQ. char_init20 ) GOTO 9920
	  ENDIF
  340	CONTINUE
	GOTO 9990

* error messages
! 9000	CALL TM_ERRMSG (merr_erreq, status, 'TM_READ_GRIDFILE',
!     .			no_descfile, no_stepfile,
!     .			'FILE='//grid_filename(newgrid),
!     .			no_errstring, *9999)

 9010	CALL TM_ERRMSG (merr_erreq, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'FILE='//grid_filename(newgrid),
     .			no_errstring, *9950)


 9050	CALL TM_ERRMSG ( merr_grfillim, status, 'TM_READ_GRIDFILE',
     .	                 no_descfile, no_stepfile,
     .	                 'MAX='//TM_STRING(DBLE(maxgridfiles)),
     .	                 no_errstring, *9999)

 9100	CALL TM_ERRMSG (merr_gridpredef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'GRIDNAME='//buff(p_gname), no_errstring,
     .			*9950)

 9200	CALL TM_ERRMSG (merr_gridlim, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(max_grids)), no_errstring,
     .			*9950)

 9300	def_error = 'LINENAME='//newline//' GRIDNAME='//buff(p_gname)
	GOTO 9399
 9310	def_error = 'LINENAME='//buff(p_lname)//' in COORD definition'
	GOTO 9399
 9320	def_error = 'LINENAME='//buff(p_lname)//' in TIME definition'
 9399	CALL TM_ERRMSG (merr_lineundef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			def_error, no_errstring, *9950)

 9400	CALL TM_ERRMSG (merr_linepredef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'LINENAME='//buff(p_lname), no_errstring,
     .			*9950)

 9500	CALL TM_ERRMSG (merr_linelim, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(max_lines)), no_errstring,
     .			*9950)

 9600	def_error = 'Read error in grid rotation field.  '//
     .	            'GRIDNAME='//buff(p_gname)
	GOTO 9699
 9610	def_error = 'Inner/outer product flag must be INNER or OUTER'//
     .	            'GRIDNAME='//buff(p_gname)
 9699	CALL TM_ERRMSG (merr_badgriddef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			def_error, no_errstring, *9950)

 9700	def_error = 'Read error in gridline length field.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
! 9710	def_error = 'Read error in gridline start field.  '//
!     .	            'LINENAME='//buff(p_lname)
!	GOTO 9799
 9720	def_error = 'Read error in gridline delta field.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9730	def_error = 'Gridline delta is negative or zero.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9740	def_error = 'Read error in initial box lower bound field.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9750	def_error = 'Read error in line coord. or box size field.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9760	def_error = 'Coord at (i) less than coord at (i-1).  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9770	def_error = 'Box size specified less than zero.  '//
     .	            'LINENAME='//buff(p_lname)
	GOTO 9799
 9780	def_error = 'A coordinate given is not enclosed by box.  '//
     .	            'LINENAME='//buff(p_lname)
 9799	CALL TM_ERRMSG (merr_badlinedef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			def_error, no_errstring, *9950)

 9800	CALL TM_ERRMSG (merr_linstorlim, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(maxlinestore)),
     .			no_errstring, *9950)

 9900	CALL TM_ERRMSG (merr_nocoordlin, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'LINENAME='//line_name(i), no_errstring,
     .			*9950)

 9910   def_error = 'Read error for line_t0 - '//buff(p_t0)
!	GOTO 9949
! 9920	def_error = 'No t0 info for line '//line_name(i)   ! disabled 9/95
 9949	CALL TM_ERRMSG (merr_badtimedef, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			def_error, no_errstring, *9950)

* Reset memory locations if error occurs
 9950	grid_filename(newgrid) = char_init
	IF (ngrid .GT. g_used) THEN
	  DO 350 i = g_used+1, ngrid
	    grid_name(i)   = char_init
  350	  CONTINUE
	ENDIF
	IF (nline .GT. l_used) THEN
	  DO 360 i = l_used+1, nline
	    line_name(i)   = char_init
	    line_subsc1(i) = unspecified_int4
  360	  CONTINUE
	ENDIF
	next_line_mem_pos = orig_line_mem_pos
	GOTO 9999

 9990	status = merr_ok

 9999	IF ( opened ) CLOSE (UNIT = lunit)
	RETURN
	END
