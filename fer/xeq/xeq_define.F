	SUBROUTINE XEQ_DEFINE( memory )

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
* execute the DEFINE command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/22/87
* revision 1.0 - 7/27/88 - added DEFINE GRID and DEFINE AXIS code
* revision 1.1 - 8/31/88 - added DEFINE VIEWPORT
* revision 1.2 -11/21/88 - new TM library: FIND_LIKE_GRID-->TM_FIND_LIKE_GRID
*					   TRANSFER_GRID --> TM_COPY_GRID
*					   READ_GRIDFILE --> TM_READ_GRIDFILE
*					         UNIT_ID --> TM_UNIT_ID
*					  FIND_LIKE_AXIS -->TM_FIND_LIKE_LINE
*                                       FIND_LINE_NUMBER --> TM_GET_LINENUM
*					      BREAK_DATE --> TM_BREAK_DATE
*			using TM_FIND_LINE_SLOT and TM_FIND_GRID_SLOT
*			and include files XUNITS and XGRID are from TMAP_FORMAT
* revision 1.3 - 1/15/89 - changed default T0 for time axes from 1-JAN-1800 to
*			   15-JAN-1901 for PPLUS WHOI format limitations
* V200:  6/29/89 - reordered lo/hi_ss arguments
*		 - added DEFINE VARIABLE
*		 - DEFINE AX: allow calendar time axis using time steps
*		 - added DEFINE AXIS/FROM_VARIABLE
*		 - DEFINE AXIS&GRID check for legal name
*		 - DEFINE VARIABLE accepts infix notation
*	 3/19/90 - fixed logic bugs when redefining axes and grids
*		 - eliminated warnings for duplicated definitions
*	 3/22/90 - don't check axis units if units are unknown
*	 4/24/90 - DEFINE VAR bug for var names of >8 chars
*	  5/2/90 - DEFINE REG/DX= name tried to work from cx_last instead of
*		   previous region "name"
* Unix/RISC 3/12/91 (incidental) LET/TITLE now preserves lower case letters
*                - DEFINE GRID/FILE=name changed to case sensitive name
* V230:   6/8/92 - added DEFINE ALIAS (bug fixes 7/29)
*         7/6/92 - added DEFINE AXIS/DEPTH (units="m" used to determine up/dn)
*        7/24/92 - bug fix: DEFINE REGION was behaving like /DEFAULT always
*                - delete uvars and uvar grids when @ regions are redefined
*       10/22/92 - added units on uvars
*       11/12/92 - added qualifiers /TEXT, /XLIMITS, /YLIMITS to DEFINE VIEW
*                  and made it auto-scale the axes
*	 3/29/93 - restore the old-style functioning of DEFINE VIEW/SIZE
* V300:  4/22/93 - fix handling of enclosing quotes on DEFINE ALIAS
* V301:	11/03/93 - added DEFINE AXIS/NPOINTS
*	11/25/93 - bug fix: DEFINE AXIS/FROM/T0= expr
*	 1/19/94 - added DEFINE VARIABLE/QUIET
* V313:  ?? *kob* - IBM port shortened a couple lines and removed a couple tabs
*	11/06/94 *sh* - inserted the bug fixes that had already been made in
*		   the checked out version of this routine as follows:
* V320: 5/94 - array "memory" as a calling argument
*	8/94 - replaced calls to TM_FIND_LINE_SLOT and TM_FIND_GRID_SLOT with
*		ALLO_MANAGED_GRID/AXIS to that deleted slots would be reused
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V400:  6/22/95 - added DEFINE SYMBOL name = value
* V420:  7/95 - Added support for  LET/D=dset var = ...
*	 11/95 - check to see if DEFINE AXIS/FROM actually has REGULAR points
*	       - default for T0 set at start of start century (not always 1900)
*	  2/96 - allow color to separate lo/high viewport limits (bug 2/12)
*	  2/96 - fixed bug introduced 11/96. Code used a non-existent variable
*		 called "regular" which was not caught by the compiler
*      2/29/96 - bug: same var name couldn't be used in multiple dsets
*	  4/96 - bug: tabs around "=" not handled right in LET A = expr
* V430:7/10/96 - allow axes with units that contradict their orientations
*		 Issue a warning
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V450:  11/96 - Using reformulated version of INIT_UVAR (RPN not called
*		 from here). Part of changes to support external function.
*	  1/97 - Issue warning if DEFINE AXIS/UNIT=MONTH is used
* $Id$ 
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
* V491+ - 7/98 *sh* - allow case-sensitive var name in DEFINE GRID/LIKE=varname

#ifdef unix
	include	'tmap_errors.parm'
#	include	"tmap_dset.parm"
	include	'tmap_dims.parm'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'rpn.parm'
	include 'gfdl_vms.parm'	
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'
	include 'xcommand.cmn'	
#else
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:RPN.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'	
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
#endif

* local parameter declarations:
	INTEGER	    slash_dflt,
     .		    slash_file,
     .		    slash_like,
     .		    slash_units,
     .		    slash_x0,
     .		    slash_t0,
     .		    slash_name,
     .		    slash_from,
     .		    slash_depth,
     .		    slash_modulo,
     .		    slash_npoints,
     .		    slash_text,
     .		    slash_xlimits,
     .		    slash_ylimits,
     .		    slash_size,
     .		    slash_origin,
     .		    slash_clip,
     .		    slash_let_title,
     .		    slash_let_units,
     .		    slash_let_quiet,
     .		    slash_let_dset
	LOGICAL	    explct_defn
	PARAMETER ( slash_dflt   = 9,
     .		    slash_file   = 1 + 4,
     .		    slash_like   = 1 + 5,
     .		    slash_units  = 1 + 5,
     .		    slash_x0     = 0,	! slash_x minus 1
     .		    slash_t0     = 1 + 6,
     .		    slash_name	 = 1 + 7,
     .		    slash_from	 = 1 + 8,
     .		    slash_depth	 = 1 + 9,
     .		    slash_modulo = 1 + 10,
     .		    slash_npoints= 1 + 11,
     .              slash_text   = 1 + 0,
     .		    slash_xlimits= 1 + 1,
     .		    slash_ylimits= 1 + 2,
     .		    slash_size   = 1 + 3,
     .		    slash_origin = 1 + 4,
     .		    slash_clip   = 1 + 5,
     .		    slash_let_title  = 1 + 0,
     .		    slash_let_units  = 1 + 1,
     .		    slash_let_quiet  = 1 + 2,
     .		    slash_let_dset   = 1 + 3,
     .		    explct_defn  = .FALSE. )

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	  TM_DIGIT, TM_LEGAL_NAME, TM_LINE_MATCH, TM_GRID_MATCH,
     .		  TM_FPEQ, dup_name, delta_given, create, irreg, its_reg
	INTEGER   REGION_NUMBER, GRID_FROM_NAME, VIEWPORT_NUMBER, STR_UPCASE,
     .		  STR_CASE_BLIND_COMPARE, CX_DIM_LEN, ALIAS_ID,
     .		  TM_GET_LINENUM, TM_LENSTR1,
     .		  TM_GET_GRIDNUM, TM_UNIT_ID,
     .		  ALLO_MANAGED_AXIS, ALLO_MANAGED_GRID, FIND_DSET_NUMBER,
     .		  tmap_status, npoints,
     .		  status, idim, vax_code, islot, iunit, iline, cat, var,
     .		  orient, mods_cx, cx, mr, n, i1, i2, i3, i4, i5, i6, pos,
     .		  grid, iqual, ax_grid, ivp, qp, iseg, uvar, dset,
     .		  old_line, old_grid, frst_pt, dim(4), ndim, num_parts, natom,
     .		  at_type  (maxatoms), at_id  (maxatoms),
     .		  at_start (maxatoms), at_end (maxatoms)
	REAL	  rbuff
	REAL*8	  SECS_FROM_BC, bc_to_t0, start, delta, end
	CHARACTER FULL_UVAR_NAME*64,
     .		  reg_name*4, buff1*32, buff2*32, buff3*40, buff30*30,
     .            basic_orients(4)*2, err_string*255, replmsg*24

        CHARACTER*1 tab
        PARAMETER     ( tab = CHAR(9) )

	DATA	  basic_orients / 'WE','SN','UD','TI' /,
     .		  replmsg/'Replacing definition of '/

* statement function - value between 0 and 1
	REAL x
	LOGICAL OUTSIDE_0_1
	OUTSIDE_0_1( x ) = x.LT.0.0 .OR. x.GT.1.0

* select subcommand
	GOTO ( 100,200,300,400,500,600,700,800 ) subcmnd_num

* DEFINE '    ' - (no action)
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser
 100	IF ( num_args .GE. 1 ) THEN
	   CALL ERRMSG( ferr_invalid_subcmnd, status,
     .			cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	ELSE
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what ?', *5000 )
	ENDIF

**************************************************************************
* DEFINE REGION
* ... command syntax ok ?
 200	IF ( num_args .EQ. 0 ) THEN
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what region ?', *5000 )
	ENDIF

* get the name of the region to be defined
	reg_name = cmnd_buff(arg_start(1):arg_end(1))

* is this region already defined ?
	islot = REGION_NUMBER( reg_name )
        create = islot .EQ. unspecified_int4

* delta context specifiers (DX=,DY=, etc), if any
	CALL GET_DELTA_CONTEXT( 10, delta_given, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

	IF ( delta_given .AND. .NOT.create ) THEN
* modify a previously defined region
* minor bug: normal non-delta location qualifiers are ignored in this case
	   CALL TRANSFER_CONTEXT( islot, cx_buff )
	   CALL APPLY_DELTA_CONTEXT( cx_buff, reg_name, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

	ELSE
* assemble the context based on the last command context
	   CALL GET_NEW_CX( cx_last, cx_buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* ... apply delta limits to the default from last command
	   IF ( delta_given ) THEN
	      CALL APPLY_DELTA_CONTEXT( cx_buff, 'DEFAULT', status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF

* ... if this region was not previously defined find a slot
	   IF ( create ) THEN
	      DO 240 islot = 0, min_context, -1
	         IF ( cx_name(islot) .EQ. unspecified_name4 ) GOTO 245!fix 7/92
 240	      CONTINUE
* ... no free slots
	      CALL ERRMSG( ferr_prog_limit,status,
     .			   'Cancel or redefine a region',*5000 )
	   ENDIF

* ... if "/DEFAULT" was implied or given then flag for saving all dimensions
*     that are defined
 245       IF ( num_qualifiers .EQ. 0 
     .     .OR. qual_given( slash_dflt ) .GT. 0 ) THEN
	      DO 210 idim = 1, 4
	         IF (cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ss(cx_buff,idim) .NE. unspecified_int4
     .	        .OR..NOT.cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ww(idim,cx_buff) .NE. unspecified_val8)
     .				cx_given( idim, cx_buff ) = .TRUE.
 210	      CONTINUE
* ... else flag only the dimensions named
	   ENDIF
	ENDIF

* all unmodified axes must be set to unspecified
	DO 230 idim = 1, 4
 230	IF ( .NOT.cx_given(idim,cx_buff) ) CALL DEL_CX_DIM( idim, cx_buff )

* LET-defined variables may involve now-redefined @regions so wipe out
* previously computed values and grids that may be obsolete(7/92)
        IF ( .NOT.create ) THEN
           CALL PURGE_ALL_UVARS
           CALL DELETE_ALL_UVAR_GRIDS
        ENDIF

* save it for posterity
 	CALL TRANSFER_CONTEXT( cx_buff, islot )
	cx_name( islot ) = reg_name
	RETURN

**************************************************************************
* DEFINE GRID
* ... DEFINE GRID/FILE=fname
 300	iqual = qual_given( slash_file )
	IF ( iqual .GT. 0 ) THEN
* ... note "err_string" is used to save space - this is not an error
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      err_string, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_READ_GRIDFILE( err_string, tmap_status )
	   IF ( tmap_status .NE. merr_ok ) GOTO 5305
	   RETURN
	ENDIF

* DEFINE GRID gridname - what is "gridname" ?
	IF ( num_args .NE. 1 ) GOTO 5300

* must be a unique name
	vax_code = STR_UPCASE( buff3, cmnd_buff(arg_start(1):arg_end(1)) ) 
	IF ( .NOT.TM_LEGAL_NAME(buff3) ) GOTO 5100
	old_grid = TM_GET_GRIDNUM( buff3 ) 
	dup_name = old_grid .NE. unspecified_int4

* was a /LIKE= given ? ... set up a grid template
	iqual = qual_given( slash_like )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff1, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   grid = GRID_FROM_NAME( buff1, cx_last, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_COPY_GRID( grid, mgrid_buff )
	ELSE
	   DO 310 idim = 1, 4
	      grid_line    (idim, mgrid_buff) = mnormal
              grid_out_prod(idim, mgrid_buff) = .TRUE.
 310	   CONTINUE	
	   grid_rotation( mgrid_buff ) = 0.0
	ENDIF

* axes named with /X=line_grid_or_var, /Y= ...
	DO 320 idim = 1, 4
	   iqual = qual_given( idim )
	   IF ( iqual .GT. 0 ) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .				 buff1, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      iline = TM_GET_LINENUM( buff1 )
	      IF ( iline .NE. unspecified_int4 ) THEN
* ... line given by name - don't allow inappropriate units
	         iunit = line_unit_code( iline )
 	         IF ( iunit.NE.0 ) THEN
 	            IF ( (  idim.EQ.t_dim .AND.
     .		     (iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time) )
     .	            .OR. (  idim.NE.t_dim .AND.
     .		     (iunit.LT.pun_1st_len  .OR. iunit.GT.pun_last_len ) )
     .	            .OR. (  idim.EQ.z_dim .AND.
     .	              iunit.EQ.pun_degrees )     ) GOTO 5330
	         ENDIF
	         grid_line(idim, mgrid_buff) = iline
	      ELSE
* ... line implied through grid or variable name
	         ax_grid = GRID_FROM_NAME( buff1, cx_last, status )
	         IF ( status .NE. ferr_ok ) RETURN
	         IF ( ax_grid .EQ. unspecified_int4 ) GOTO 5320
	         grid_line(idim, mgrid_buff) = grid_line( idim, ax_grid )
	      ENDIF
	   ENDIF
 320	CONTINUE

* check for repeated definitions and illegal redefinitions
	IF ( dup_name ) THEN
	   IF ( TM_GRID_MATCH( old_grid,mgrid_buff ) ) THEN
	      RETURN					! dup name/dup def
	   ELSE
* ... redefinition makes all past assumptions incorrect
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			replmsg//'grid '//grid_name(old_grid), 0)
	      CALL PURGE_MR_GRID( old_grid, status )
	      IF ( status .NE. merr_ok ) GOTO 5000
	      grid_name( old_grid ) = char_init16
	   ENDIF
	ENDIF

* find a slot to store the new grid
	status = ALLO_MANAGED_GRID( grid )
	IF ( status .NE. merr_ok ) GOTO 5000

* save the grid and name it
 340	CALL TM_COPY_GRID( mgrid_buff, grid )
	grid_name( grid ) = buff3
	RETURN

**************************************************************************
* DEFINE VARIABLE
* find name to define
* 2 syntaxes are allowed: "DEFINE VAR NAME=TEXT", "DEFINE VAR NAME TEXT"
* in the latter be careful that "TEXT" may contain "="
 400	IF ( num_args .LT. 1 ) GOTO 5300
	pos = INDEX(cmnd_buff(arg_start(1):len_cmnd), '=')
* ... is it a legit = sign or one used within [] further on in expression ?
	IF ( num_args .GT. 1 ) THEN
	   IF ( pos+arg_start(1)-1 .GT. arg_start(2) ) pos = 0
	ENDIF
	IF ( pos .EQ. 0 ) THEN
* ... no "=" separator
	   IF ( num_args .LT. 2 ) GOTO 5400
	   buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	   pos = arg_start(2)
	ELSE
* ... yes "=" separator
	   pos = pos + arg_start(1)	! character after "="
*    ... hunt backwards for end of variable name
	   buff1 = '(none)'
	   DO 401 i1 = pos-2, arg_start(1), -1
	      IF ( cmnd_buff(i1:i1).EQ.' '
     .	      .OR. cmnd_buff(i1:i1).EQ.tab ) THEN
	         CONTINUE
	      ELSE
	         GOTO 402
	      ENDIF
 401	   CONTINUE
	   GOTO 5410
 402	   buff1 = cmnd_buff( arg_start(1):i1 )
*    ... find next non-blank, non-tab after "=" -- start of expression
	   DO 405 pos = pos, len_cmnd
	      IF ( cmnd_buff(pos:pos) .NE. ' ' 
     .	     .AND. cmnd_buff(pos:pos) .NE. tab  ) GOTO 406
 405	   CONTINUE
	   GOTO 5400
 406	   CONTINUE
	ENDIF
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410
        IF ( TM_LENSTR1(buff1) .GT. 24 ) GOTO 5410
	CALL FIND_VAR_NAME ( pdset_irrelevant, buff1, cat, var )
	IF (  var .NE. munknown_var_name
     .	.AND. cat .EQ. cat_pseudo_var    ) GOTO 5420

* /D=xxx: has the user requested a data-set-specific definition?
	iqual = qual_given( slash_let_dset )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff3 .EQ. ' ' ) THEN
	      dset = pdset_irrelevant	! "/D", alone for default-to-dset
	      buff3 = 'No current data set'	! ... for error reporting
	      IF ( dset .EQ. unspecified_int4) GOTO 5470
	   ELSE
	      dset = FIND_DSET_NUMBER( buff3 )
	      IF ( dset .EQ. unspecified_int4) GOTO 5470
	   ENDIF
	ELSE
	   dset = unspecified_int4
	ENDIF

* title given ?  (preserves lower case via EQUAL_STR_LC 3/91)
	buff3 = ' '
	iqual = qual_given( slash_let_title )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* units given ?
	buff2 = ' '
	iqual = qual_given( slash_let_units )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* break the expression into atoms and identify them
	CALL ALG_BREAK_UP( cmnd_buff(pos:len_cmnd),at_start,at_end,natom,
     .			   err_string,status )
	IF ( status .NE. ferr_ok ) GOTO 5440
	CALL ALG_ID( cmnd_buff(pos:len_cmnd), natom, at_start, at_end,
     .		     at_type, at_id, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* if this definition replaces a variable of the same name in this data set
* then purge cached calculations (8/95)
	IF ( dset .GT. pdset_irrelevant ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
	   IF ( cat .EQ. cat_file_var
     .	  .AND. var .NE. munknown_var_name ) CALL PURGE_ALL_UVARS
	ENDIF

* if this is a global forground definition that replaces any variable
* then purge cached calculations (8/95)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
	   IF ( var .NE. munknown_var_name ) CALL PURGE_ALL_UVARS
	ENDIF

* now file the expression in the user-defined variable area
	CALL INIT_UVAR ( buff1, cmnd_buff(pos:len_cmnd), buff3, buff2, dset,
     .			 explct_defn, uvar,
     .			 at_type, at_id, at_start, at_end,
     .			 natom, 1, len_cmnd-pos+1, status )
	IF ( status .NE. ferr_ok ) RETURN

* do we need to delete a previous definition of this var ?
* Note: need to purge old defs ALSO if the same name is now redefined in a 
*	new data set -- expressions using this name chg interpretation  
	DO 410 i1 = 1, max_uvar
	   IF ( i1 .EQ. uvar
     .	   .OR. uvar_num_items(i1) .EQ. uvar_deleted ) GOTO 410
	   vax_code = STR_CASE_BLIND_COMPARE( uvar_name_code(i1), buff1 )
	   IF ( vax_code .EQ. vms_str_success ) THEN
* ... delete only if same name in same data set
	      IF ( uvar_dset(i1) .EQ. dset ) THEN
	         uvar_num_items(i1) = uvar_deleted
	         IF ( qual_given(slash_let_quiet) .EQ. 0 ) THEN
	            err_string = FULL_UVAR_NAME(i1,i2)	! buffer
	            CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			replmsg// 'variable '//err_string(:i2), 0)
	         ENDIF
	      ENDIF
* ... reuse of name  makes all past assumptions incorrect
	      CALL PURGE_ALL_UVARS	! not done above for pdset_irrelevant
!	      GOTO 420   ! no GOTO: new name may match more than 1 var
	   ENDIF
 410	CONTINUE

* is there at least one slot blank ?
 420	DO 430 i1 = 1, max_uvar
 430	IF ( uvar_num_items(i1) .EQ. uvar_deleted ) RETURN

* no - not allowed to fill the last slot !
	uvar_num_items(uvar) = uvar_deleted
	CALL ERRMSG( ferr_prog_limit, status,
     .		'too many user-defined variables'//pCR//
     .		'cancel or redefine some variables', *5000 )

**************************************************************************
* DEFINE AXIS
* ... /FILE=
 500	IF ( qual_given( slash_file ) .GT. 0 ) GOTO 300 ! like DEFINE GRID/FILE

* DEFINE AXIS - what is "axisname" ?
	iqual = qual_given( slash_name )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSE	   
	   IF ( num_args .NE. 1 ) GOTO 5300
	   vax_code = STR_UPCASE( buff3, cmnd_buff(arg_start(1):arg_end(1)) ) 
	ENDIF

* must be a unique name
	IF ( .NOT.TM_LEGAL_NAME( buff3 ) ) GOTO 5100
	old_line = TM_GET_LINENUM( buff3 )
	dup_name = old_line .NE. unspecified_int4

* get the X,Y,Z,T regions
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS( mods_cx )
	CALL GET_CONTEXT_MODS (	cx_last,
     .				cmnd_buff,
     .				mods_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				unknown_qual_ok,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5000

* orientation must be uniquely defined
	orient = unspecified_int4
	DO 510 idim = 1, 4
	   IF ( qual_given(slash_x0+idim) .GT. 0 ) THEN
	      IF ( orient .NE. unspecified_int4 ) GOTO 5510
	      orient = idim
	   ENDIF
 510	CONTINUE
        IF ( qual_given( slash_depth ) .GT. 0 
     . .AND. orient .NE. z_dim            ) GOTO 5505
	IF ( orient .EQ. unspecified_int4 ) GOTO 5510

* release context stack space
	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* /UNITS = DEGREES,METERS, ... SECONDS (or supply defaults)
	iqual = qual_given( slash_units )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff1, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSEIF ( orient .EQ. t_dim
* 8/97 *kob* use cx_calender to check if time is date rather than time step
     .  .AND.  (     qual_given(slash_T0) .GT. 0
     .		.OR. cx_calendar(mods_cx))) THEN
	   buff1 = 'HOURS'
	ELSE
	   buff1 = 'NONE'
	ENDIF

* interpret the line units
	IF ( buff1 .EQ. 'NONE' .OR. buff1 .EQ. ' ' ) THEN
	   buff1 = ' '
	   iunit = 0
	ELSE
	   iunit = TM_UNIT_ID( buff1 )
	   IF ( iunit .EQ. 0 ) THEN
	       CALL WARN('unknown axis units: '//buff1)
	   ELSEIF (iunit .EQ. -5) THEN    ! 1/97: "-5" is "MONTHS"
	         CALL WARN
     .		('/UNIT=MONTHS is ambiguous ... using 1/12 of 365 days.') 
	   ELSE
* ... Issue warning for inappropriate units like HOURS on X ax
*     and render the units unconvertible
 	      IF ( (  orient.EQ.t_dim .AND.
     .		   ( iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time ) )
     .	      .OR. (  orient.NE.t_dim .AND.
     .		   ( iunit.LT.pun_1st_len  .OR. iunit.GT.pun_last_len  ) )
     .	      .OR. (  orient.EQ.z_dim .AND.
     .	             iunit.EQ.pun_degrees )  ) THEN	! 7/96
	         CALL WARN
     .		   ('Units appear to conflict with orientation:'
     .			//' no conversions possible')
	         iunit = 0
	      ENDIF	      
	   ENDIF
 	ENDIF

* find a space to catalog it
	status = ALLO_MANAGED_AXIS( iline )
	IF ( status .NE. merr_ok ) GOTO 5000

* was /NPOINTS given ?
	iqual = qual_given( slash_npoints )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_val( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      rbuff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( rbuff .LE. 0.0 .OR. rbuff .GT. 1E6 ) GOTO 5582
	   npoints = rbuff
	ELSE
	   npoints = unspecified_int4
	ENDIF

* GET THE AXIS POINTS: irregular (from_variable) or regular
	irreg = qual_given(slash_from) .GT. 0
	IF ( irreg ) THEN
* ... get axis points from a memory variable
	   CALL GET_CMND_DATA ( memory, cx_last, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( num_uvars_in_cmnd .NE. 1 ) GOTO 5580
	   cx = is_cx(1)
	   mr = is_mr(1)
* ... make sure it's an line (not a point,plane,etc.) of data
	   CALL GET_CX_DIMS( cx, ndim, dim )
	   IF ( ndim .GT. 1 ) GOTO 5580
	   idim = dim(1)
	   n    = CX_DIM_LEN( idim, cx )
* ... original start of free line memory
	   frst_pt = next_line_mem_pos
* ... allocate some storage for the coordinates
	   IF (frst_pt+2*n .GT. maxlinestore) GOTO 5590
* ... store coordinate positions (error status already checked)
	   CALL NON_ARRAY_SUBSC( mr, 1 )
	   CALL EXTRACT_DP_LINE( cx, memory(1, mr_blk1(mr)),
     .				 line_mem(frst_pt), idim, n )
	   IF ( npoints .EQ. unspecified_int4 ) npoints = n
* ... store initial box lower bound (start of axis)
	   line_mem(frst_pt+npoints) = line_mem(frst_pt) -
     .			0.5 * ( line_mem(frst_pt+1) - line_mem(frst_pt) )
* ... mid-axis box boundaries ...
	   DO 525 i1 = 1, npoints-1
	      i2 = frst_pt + i1
	      IF ( line_mem(i2-1) .GE. line_mem(i2) ) GOTO 5570
	      line_mem(i2+npoints) = ( line_mem(i2-1) + line_mem(i2) ) * 0.5
 525	   CONTINUE
* ... upper-most box bound (end of axis)
	   i2 = frst_pt+npoints-1	! last box point
	   line_mem(i2+npoints+1) = line_mem(i2) +
     .			0.5 * ( line_mem(i2) - line_mem(i2-1) )
* ... check to see if it was regularly spaced after all
	   its_reg = .TRUE.	! until proven otherwise
	   start = line_mem(frst_pt)
           delta = line_mem(frst_pt+1) - start
           DO 550 i1 = frst_pt+2,frst_pt+npoints-1
              IF (.NOT.TM_FPEQ( SNGL(delta),
     .               SNGL(line_mem(i1)-line_mem(i1-1))) )
     .						its_reg = .FALSE.
 550	   CONTINUE
	   IF ( its_reg ) THEN
	      line_start    ( iline ) = start
	      line_delta    ( iline ) = delta
	      line_regular  ( iline ) = .TRUE.
	      line_subsc1   ( iline ) = unspecified_int4
	   ELSE	      
* ... save irregular line data
	      line_subsc1 ( iline ) = frst_pt
	      line_start  ( iline ) = line_mem(frst_pt)
	      line_delta  ( iline ) = unspecified_val8
	      line_regular( iline ) = .FALSE.
	   ENDIF
	ELSE

* ... start,end,delta specification for a regular axis
	   start = cx_lo_ww(orient,mods_cx)
	   end   = cx_hi_ww(orient,mods_cx)
	   IF ( npoints .EQ. unspecified_int4 ) THEN
	      delta = cx_delta (orient,mods_cx)
	      IF ( delta.EQ.unspecified_val8 .OR. delta.LE.0.D0) GOTO 5520
* 9/97 *kob* - check for negative time axis using cx_calendar
*	      IF ( .NOT.(orient.EQ.t_dim .AND. start.LT.0.0D0) )
	      IF ( .NOT.(orient.EQ.t_dim .AND. cx_calendar(mods_cx)) )
     .			npoints = ( (end - start) / delta ) + 1.9999
	   ELSEIF ( npoints .EQ. 1 ) THEN
	      delta = 1.D0
c	   ELSEIF ( orient.EQ.t_dim .AND. start.LT.0.0D0 ) THEN
	   ELSEIF ( orient.EQ.t_dim .AND. cx_calendar(mods_cx) ) THEN
	      delta = (start - end) / ( (npoints-1)*un_convert(iunit) )
	      IF ( delta .LE. 0.0D0 ) GOTO 5520
	   ELSE
	      delta = (end - start) / (npoints-1)
	      IF ( delta .LE. 0.0D0 ) GOTO 5520
	   ENDIF
	   line_start    ( iline ) = start
	   line_delta    ( iline ) = delta
	   line_regular  ( iline ) = .TRUE.
	   line_subsc1   ( iline ) = unspecified_int4
	ENDIF

* /T0 = date (or supply default)
	iqual = qual_given( slash_T0 )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_BREAK_DATE( buff2, i1,i2,i3,i4,i5,i6, status )
	   IF ( buff2 .EQ. ' ' .OR. status .NE. ferr_ok ) GOTO 5540
	ELSE
	   buff2 = '15-JAN-1901'		! default
* 11/95 - shift the default T0 to a value that is early enough
*	   IF ( orient.EQ.t_dim .AND. start.LT.0.0D0 ) THEN 
	   IF ( orient.EQ.t_dim .AND. cx_calendar(mods_cx) ) THEN
	      i1 = 20
 560	      bc_to_t0 = SECS_FROM_BC( buff2, status )      
	      IF ( status .NE. ferr_ok ) GOTO 5540
	      IF ( bc_to_t0 + start .GE. 0 ) THEN	! start is before T0?
	         IF ( i1 .LE. 0 ) GOTO 5540
	         i1 = i1 - 1
	         WRITE (buff2,3560) i1
 3560	         FORMAT('01-JAN-',I2.2,'00')
	         GOTO 560
	      ENDIF
	   ENDIF
	ENDIF

* save the line in memory - but don't catalog its name
	line_dim      ( iline ) = npoints
	line_units    ( iline ) = buff1	   
	line_t0       ( iline ) = buff2
	line_unit_code( iline ) = iunit
	line_direction( iline ) = 'NA'
	IF ( orient .EQ. t_dim ) line_tunit(iline) = un_convert( iunit )

* is it a modulo axis ?
        line_modulo( iline ) = qual_given(slash_modulo) .GT. 0

* is it a formattable axis ?
	IF ( orient .EQ. x_dim .AND. iunit .EQ. pun_degrees ) THEN
	   line_direction( iline ) = 'WE'

	ELSEIF ( orient .EQ. y_dim .AND. iunit .EQ. pun_degrees ) THEN
	   line_direction( iline ) = 'SN'

	ELSEIF ( orient .EQ. z_dim ) THEN
           IF ( qual_given( slash_depth ) .GT. 0 ) THEN
	      line_direction( iline ) = 'UD'    ! GEOG_LABEL requires "meters"
           ELSE
	      line_direction( iline ) = 'DU'
           ENDIF

* ... time axis specified as date strings  (ELSEIF mod 11/93 *sh*)
	ELSEIF ( .NOT.irreg
c     .	    .AND. orient .EQ. t_dim .AND. start .LT. 0.0D0 ) THEN
     .	    .AND. orient .EQ. t_dim .AND. cx_calendar(mods_cx) ) THEN
	   line_direction(iline) = 'TI'
	   bc_to_t0 = SECS_FROM_BC( line_t0(iline), status )
	   line_dim  (iline) = (start-end) / (line_tunit(iline)*delta) + 1.9999
	   line_start(iline) = (-line_start(iline)-bc_to_t0)/line_tunit(iline)
	   IF ( line_start(iline) .LE. 0.0D0 ) GOTO 5360

* ... time axis specified as time step values with T0 given
	ELSEIF ( orient.EQ.t_dim .AND. qual_given(slash_T0).GT.0 ) THEN
	   line_direction(iline) = 'TI'
	ENDIF

* check for redefinitions
	IF ( dup_name ) THEN
	   IF ( TM_LINE_MATCH( old_line, iline ) ) THEN
	      RETURN					! dup name/dup def
	   ELSE
* ... redefinition makes all past assumptions incorrect
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			replmsg//'axis '//line_name( old_line ), 0)
	      CALL PURGE_MR_AXIS( old_line, iline, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ENDIF

* now catalog it in memory by saving its name
	line_name( iline ) = buff3
* ... coordinate pointer for next line to save
	IF (qual_given(slash_from).GT.0 .AND. .NOT.line_regular(iline))
     .			next_line_mem_pos = frst_pt + 2*n + 1
	RETURN

**************************************************************************
* DEFINE VIEWPORT
 600	IF ( num_args .NE. 1 ) GOTO 5300
* check out viewport name
	buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	ivp = VIEWPORT_NUMBER( buff1 )
	IF ( ivp .EQ. unspecified_int4 ) THEN
* ... new name - find an unused slot
	   DO 610 ivp = 1, max_viewport
 610	   IF ( vp_name(ivp) .EQ. unspecified_name4 ) GOTO 620
	   GOTO 5610	! no free slots
	ELSE
* ... name in use - flag segments in this viewport as deleted
	   IF ( ivp .EQ. mvp_dflt ) GOTO 5640
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			'Re-defining viewport '//vp_name(ivp), 0)
	   DO 614 iseg = vp_seg0(ivp)+1 , vp_segn(ivp)
 614	   CALL GDSG( iseg ) 
	   vp_seg0 ( ivp ) = 0
	   vp_segn ( ivp ) = 0
	ENDIF

* pre-set all values to defaults
 620	vp_size ( ivp ) = 1.0
	vp_xorg ( ivp ) = 0.0
	vp_yorg ( ivp ) = 0.0
	vp_xclip( ivp ) = unspecified_val4   ! old style  3/29/93
	vp_yclip( ivp ) = unspecified_val4   ! old style  3/29/93
	vp_seg0 ( ivp ) = 0
	vp_segn ( ivp ) = 0

***************** OLD CODE (retained for greater upwards compatibility)
* /ORIGIN = x,y
	qp = qual_given( slash_origin )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.TM_DIGIT(buff2) .OR. INDEX(buff2,',').EQ.0 ) GOTO 5630
	   READ ( buff2, *, ERR=5630 ) vp_xorg(ivp), vp_yorg(ivp)
	   IF ( OUTSIDE_0_1(vp_xorg(ivp)) .OR. OUTSIDE_0_1(vp_yorg(ivp)) )
     .	      							   GOTO 5620
	ENDIF

* /SIZE=q
	qp = qual_given( slash_size )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   vp_size(ivp), status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( vp_size(ivp) .LT. 0.0 ) GOTO 5620
           vp_size(ivp) = -1. * vp_size(ivp)    ! flag for old-style viewports
           CALL WARN (
     .          'The /SIZE qualifier will force Ver. 2.2 behavior'
     .	        //' for DEFINE VIEWPORT')
	ENDIF

* /CLIP = x,y
	qp = qual_given( slash_clip )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.TM_DIGIT(buff2) .OR. INDEX(buff2,',').EQ.0 ) GOTO 5630
	   READ ( buff2, *, ERR=5630 ) vp_xclip(ivp), vp_yclip(ivp)
	   IF ( vp_xclip(ivp).LE.vp_xorg(ivp) .OR. vp_xclip(ivp).GT.1.0
     .	   .OR. vp_yclip(ivp).LE.vp_yorg(ivp) .OR. vp_yclip(ivp).GT.1.0 )
     .	      							   GOTO 5620
	ENDIF

***************** END OF OLD CODE (retained for greater upwards compatibility)

* default clipping at 1,1 in new style viewports  (immaterial, I think ...)
	IF ( qual_given(slash_size) .EQ. 0 ) THEN
	   IF ( vp_xclip(ivp).EQ.unspecified_val4) THEN
	      vp_xclip(ivp) = 1.0
	      vp_yclip(ivp) = 1.0
	   ENDIF
	ENDIF

* /XLIMITS = xlow,xhigh 
	qp = qual_given( slash_xlimits )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   i1 = TM_LENSTR1(buff2)
	   pos = MAX( INDEX(buff2,','), INDEX(buff2,':') )
	   IF ( pos.LE.1 .OR. pos.EQ.i1 ) GOTO 5630
	   IF (.NOT.TM_DIGIT(buff2) )  GOTO 5630
	   READ ( buff2(1:pos-1), *, ERR=5630 ) vp_xorg(ivp)
	   IF (.NOT.TM_DIGIT(buff2(pos+1:i1))  ) GOTO 5630
	   READ ( buff2(pos+1:i1), *, ERR=5630 ) vp_xclip(ivp)
	   IF ( OUTSIDE_0_1(vp_xorg(ivp))
     .     .OR. OUTSIDE_0_1(vp_xclip(ivp))
     .     .OR. vp_xorg(ivp) .GE. vp_xclip(ivp) )  GOTO 5620
	ENDIF

* /YLIMITS = ylow,yhigh 
	qp = qual_given( slash_ylimits )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   pos = MAX( INDEX(buff2,','), INDEX(buff2,':') )
	   i1 = TM_LENSTR1(buff2)
	   IF ( pos.LE.1 .OR. pos.EQ.i1 ) GOTO 5630
	   IF (.NOT.TM_DIGIT(buff2) )  GOTO 5630
	   READ ( buff2(:pos-1), *, ERR=5630 ) vp_yorg(ivp)
	   IF (.NOT.TM_DIGIT(buff2(pos+1:i1))  ) GOTO 5630
	   READ ( buff2(pos+1:i1), *, ERR=5630 ) vp_yclip(ivp)
	   IF ( OUTSIDE_0_1(vp_yorg(ivp))
     .     .OR. OUTSIDE_0_1(vp_yclip(ivp))
     .     .OR. vp_yorg(ivp) .GE. vp_yclip(ivp) )  GOTO 5620
	ENDIF

* /TEXT=q 
	qp = qual_given( slash_text )
	IF ( qp .GT. 0 ) THEN
	   IF ( qual_given(slash_size) .GT. 0 ) GOTO 5650
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   vp_size(ivp), status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( vp_size(ivp) .LT. 0.001 ) GOTO 5620
	ELSEIF ( qual_given(slash_size) .EQ. 0 ) THEN     ! auto-sizing
           vp_size(ivp) = SQRT( (vp_xclip(ivp)-vp_xorg(ivp))
     .                        * (vp_yclip(ivp)-vp_yorg(ivp)) )
	ENDIF

* successfully defined viewport
	vax_code = STR_UPCASE( vp_name(ivp), buff1 )

* if re-defining the current viewport then it takes immediate effect
	IF ( ivp .EQ. vp_num ) CALL SET_VIEWPORT( ivp )
	RETURN

**************************************************************************
* DEFINE ALIAS
 700	IF ( num_args .LT. 1 ) GOTO 5300
	pos = INDEX(cmnd_buff(arg_start(1):len_cmnd), '=')
* ... is it a legit = sign or one used within [] further on in expression ?
	IF ( num_args .GT. 1 ) THEN
	   IF ( pos+arg_start(1)-1 .GT. arg_start(2) ) pos = 0
	ENDIF
	IF ( pos .EQ. 0 ) THEN
* ... no "=" separator
	   IF ( num_args .LT. 2 ) GOTO 5400
	   buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	   pos = arg_start(2)
	ELSE
* ... yes "=" separator
	   pos = pos + arg_start(1)	! character after "="
	   buff1 = cmnd_buff( arg_start(1):pos-2 )
* ... find next non-blank
	   DO 705 pos = pos, len_cmnd
	      IF ( cmnd_buff(pos:pos) .NE. ' ' ) GOTO 706
 705	   CONTINUE
	   GOTO 5400
 706	   CONTINUE
	ENDIF
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* find a slot to save this alias
        i1 = ALIAS_ID( buff1 )
        IF ( i1 .EQ. unspecified_int4 ) THEN
	   DO 710 i1 = num_predefined_alias+1, total_num_alias
	      IF ( alias_name(i1) .EQ. unspecified_name4 ) GOTO 720
 710	   CONTINUE
           GOTO 5710
        ENDIF

* save the alias
 720    vax_code = STR_UPCASE( alias_name(i1), buff1 ) 
        alias(i1) = cmnd_buff(pos:arg_end(num_args))
        alias_len(i1) = TM_LENSTR1( alias(i1) )
	RETURN


**************************************************************************
* DEFINE SYMBOL
 800	IF ( num_args .LT. 1 ) GOTO 5300
	pos = INDEX(cmnd_buff(arg_start(1):len_cmnd), '=')
* ... is it a legit = sign or one used within [] further on in expression ?
	IF ( num_args .GT. 1 ) THEN
	   IF ( pos+arg_start(1)-1 .GT. arg_start(2) ) pos = 0
	ENDIF
	IF ( pos .EQ. 0 ) THEN
* ... no "=" separator
	   IF ( num_args .LT. 2 ) GOTO 5400
	   buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	   pos = arg_start(2)
	ELSE
* ... yes "=" separator
	   pos = pos + arg_start(1)	! character after "="
	   buff1 = cmnd_buff( arg_start(1):pos-2 )
* ... find next non-blank
	   DO 805 pos = pos, len_cmnd
	      IF ( cmnd_buff(pos:pos) .NE. ' ' ) GOTO 806
 805	   CONTINUE
	   GOTO 5400
 806	   CONTINUE
	ENDIF
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* if this symbol already exists then delete it
        buff30 = buff1	! must be exactly 30 characters for PLOT+ compat
	CALL GETSYM( buff30, err_string, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(buff30,i1)

* define the symbol
	err_string = cmnd_buff(pos:arg_end(num_args))	! exactly 255 for PPL
	CALL PUTSYM( buff30, err_string, arg_end(num_args)-pos+1, i2)
! ... no check on returned error code (could be "internal error")
	RETURN

* error exit
 5000	RETURN
 5100 	CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff3,*5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE what name?', *5000 )
 5305	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5310	CALL ERRMSG( ferr_grid_definition, status,
     .		'name already in use:'//buff3(:TM_LENSTR1(buff3)), *5000 )
 5320	CALL ERRMSG( ferr_grid_definition, status,
     .		'unknown grid or axis:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5330	CALL ERRMSG( ferr_grid_definition, status,
     .		'inappropriate orientation for '//ww_dim_name(idim)//
     .		' axis:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5360	CALL ERRMSG( ferr_grid_definition, status,
     .		'time axis start date must be after T0='//line_t0(iline),
     .		*5000 )
 5400 	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(:len_cmnd)//' ???', *5000 )
 5410   CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff1,*5000 )
 5420   CALL ERRMSG( ferr_syntax,status,'cannot define: '//buff1,*5000 )
 5440	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(pos:len_cmnd)//pCR//err_string, *5000 )
 5470	CALL ERRMSG( ferr_unknown_data_set, status,
     .			buff3(:TM_LENSTR1(buff3)), *5000 )
 5505	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE AXIS/DEPTH needs /Z=lo:hi:del', *5000 )
 5510	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE AXIS needs /X,/Y,/Z or /T orientation', *5000 )
 5520	CALL ERRMSG( ferr_grid_definition, status,
     .			'error in start,end,delta', *5000 )
 5540	CALL ERRMSG( ferr_grid_definition, status,
     .		'/T0='//buff2(:TM_LENSTR1(buff2)), *5000 )
 5550	CALL ERRMSG( ferr_grid_definition, status,
     .		'inappropriate units:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5570	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is not monotonic', *5000 )
 5580	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is ambiguous', *5000 )
 5582	CALL ERRMSG( ferr_grid_definition, status,
     .		'illegal agrument for DEFINE AXIS/NPOINTS', *5000 )
 5590	CALL ERRMSG( ferr_prog_limit, status,
     .		'axis coordinate storage exhausted - restart FERRET', *5000 )
 5610	CALL ERRMSG( ferr_prog_limit, status,
     .		'redefine or cancel an existing viewport', *5000 )
 5620 	CALL ERRMSG( ferr_out_of_range, status,
     .			cmnd_buff(qual_start(qp):qual_end(qp)), *5000 )
 5630 	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(qual_start(qp):qual_end(qp)), *5000 )
 5640 	CALL ERRMSG( ferr_invalid_command, status,
     .			'cannot re-define FULL viewport', *5000 )
 5650 	CALL ERRMSG( ferr_syntax, status,
     .	'DEFINE VIEWPORT/SIZE and /TEXT are mutually exclusive', *5000 )

 5710	CALL ERRMSG( ferr_prog_limit, status,
     .		'use CANCEL ALIAS to reclaim space', *5000 )

	END

