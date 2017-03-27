	SUBROUTINE GET_PROT_CMND_DATA (  dflt_cx, dtype, status )

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
* "get protected command data"
* calculate grid(s) of data based on the expression given or implied.
* Expression(s) are in user-variable (uvar) structures.
* Implied expression is left from last command
* when done - return with the variables still "protected" (flagged as in use)

* a sample command may look like:
*	prompt> CONTOUR/Z=50 TEMP[X=160E:160W]
* the "Z=50"		is a "command  qualifier" or "command  context modifier"
* the "X=160E:160W"	is a "variable qualifier" or "variable context modifier"

* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230: 7/27/92 - new routine based on GET_CMND_DATA with the unprotecting
* of variables removed to a separate routine
*  (revision history for GET_CMND_DATA shown here)

*   rev 0.0 - 6/17/86
*   rev 1.0 - 1/5/87  - multiple variables in a single expression may be
*			   evaluated and pointers returned in COMMON/XCONTEXT
*   rev 2.0 - 1/5/87  - evaluations done on the INTERPRETATION STACK.  User-
*			   defined variables introduced
*   rev 2.1 - 9/16/87 - added get_cx_only to calling sequence
*			   and unknown_qual_ok moved to FERRET.PARM
*   rev 2.2 - 9/22/87 - shortened code using GET_NEW_cx
*   rev 2.3 - 9/30/87 - added mr_perm_prot
*   rev 2.4 - 7/26/88 - don't pass unstand_grid to last context
*   V200:  5/18/89 - don't pass context on to next command
*	  	   - initialize grid stack pointer
*          8/19/91 - explicit SAVE statement for last expression buffer
* V312: 5/94 - array "memory" as a calling argument
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA statment rather than a
*	        PARAMETER statement because laslen was modified in code
*	      - Linux F90 doesn't allow SAVE statement, so put an ifdef
*		check around it 
* V500:  1/99 - allowing strings in quoble quotes requires fix to quote
*		handling as in LIST "hello"
*		Added call to ALL_1_ARG
* V530: *sh* 9/00 - added data type ad an argument
*		  - increased size of "last" stored command
* V540: *sh* 11/00 - "last" var init moved to DELETE_OLD_EXPR
* v552: *acm* 4/03   Increase length of variable names to 128 characters
* V600 *acm* 3/06  fixing bugs 439&1390, new arg to get_new_cx.
* V690 *sh* 12/13  manage auto-curvilinear mode
*	           aux vars returned on the ISP stack in frame above is_aux_ptr
* V693+ *sh* 2/15  eliminate inconsistent treatment of surrounding quotes
*                  as seen in err694_inconsistent_quotes.jnl
* V698 *acm*  3/16 Ticket 2278: add an attribute with time range when
*                  saving a modulo-regridded variable to netCDF
* V702 *acm* 11/16 Ticket 2489: Do not try to add the attribute when
*                  it's an expression instead of a user-defined variable
*		   - And trac enhancement #2369 -- dynamic memory management 
* V702 *sh*  3/17 Dynamic memory

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'interp_stack.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'
        include 'xtra_compute_attr.cmn'
        include 'netcdf.inc'

* calling argument declarations
	INTEGER	dflt_cx, dtype, status

* local variable declarations
	LOGICAL		auto_auxvar_changed
	INTEGER		TM_LENSTR1, STR_SAME, nu, cx_expr, idim
	CHARACTER	EXPR_NAME*8, VAR_CODE*128, buff*128

	INTEGER		cat, var, dset, varid, attid, attype, attlen, attoutflag
	REAL		dummy
	CHARACTER SANITARY_VAR_CODE*128, varcode*128

* initialize grid stack pointer (pushes down from top)
	grd_stk_ptr = max_grids

* initialize aux stack frames at top of isp stack
	max_usable_intrp = max_intrp
	max_context = max_noframe_context  ! just to be clear (not needed)

* initialize memory usage diagnostic
	completed_args_mem = 0
	nsplits = 0

* determine if in auto-curvilinear mode
	CALL SET_AUTO_CURVI_MODES( auto_auxvar_changed )

* delete any memory left-overs from last command
	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, .TRUE., status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* consolidate arguments broken up by command parser back into a single arg
* ... correct mis-handling of surrounding quotation marks
	CALL ALL_1_ARG_EXPR

* * * * BREAK FULL EXPRESSION INTO SUB_EXPRESSIONS * * *
* ... and store them in the uvar list names as "EX#n"
* ... eg. LIST U^2,V^2 breaks into "U^2" and "V^2"
* ... or re-use the last expression ?
	IF ( num_args .GT. 0 
     .  .OR. auto_auxvar_changed ) THEN
	   IF ( last_cmnd_buff(:len_last_cmnd)
     .     .NE.      cmnd_buff(arg_start(1):arg_end(1)) 
     .	   .OR. num_uvars_in_cmnd .EQ. cmnd_uvars_not_given
     .	   .OR. auto_auxvar_changed 		) THEN
	      CALL DELETE_OLD_EXPR
	      last_cmnd_buff = cmnd_buff(arg_start(1):arg_end(1))
	      len_last_cmnd  = arg_end(1) - arg_start(1) + 1
	      CALL BREAK_UP_EXPR( cmnd_buff(arg_start(1):arg_end(1)),
     .				  num_uvars_in_cmnd, status )
	      IF ( status .NE. ferr_ok ) THEN
	         num_uvars_in_cmnd = cmnd_uvars_not_given
	         GOTO 5000
	      ENDIF
	   ENDIF
	ENDIF

* * * * EVALUATE THE SUB-EXPRESSIONS * * *
	IF ( num_uvars_in_cmnd .EQ. cmnd_uvars_not_given )
     .		CALL ERRMSG
     .			(ferr_invalid_command,status,'no data specified',*5000)

	DO 300 nu = 1, num_uvars_in_cmnd

	   CALL RESET_CLIMAT_ATT

* ... a context slot to hold a fresh command context for each sub-expression
	   cx_stack_ptr = cx_stack_ptr + 1
	   cx_expr = cx_stack_ptr


! *sh* 1/17
! This block of code is experimental / untested from 2014.
! It is connected to  CALL SET_AUTO_CURVI_MODES above (also half-baked).
! For now I have commented it out, as it can eat up the context stack on
! commands where num_uvars_in_cmnd is large
! The logic here reserves a frame of nferdims slots at the top of the context
! stack as the place to stash the contexts needed to describe the auxiliary
! variables that are associated with each requested variable.
! e.g. SHADE my_curvi_var should automate a 3-arg curvilinear plot.
!  and LIST my_curvi_var should automatically list curvilinear coordinates
! The stack frames detected by ITS_AUTO_CURVI (also temporarily disabled)
! The code development only got as far as XEQ_LIST, (not SHADE, etc.) and
! the relevant code is never triggered as long as ITS_AUTO_CURVI returns FALSE
!* ... a frame of size nferdims to hold aux vars needed for plotting and listing
!	   max_usable_intrp = max_usable_intrp - nferdims
!	   max_context      = max_context      - nferdims
!	   DO idim = 1, nferdims
!	      is_mr(max_usable_intrp+idim) = unspecified_int4
!	      is_cx(max_usable_intrp+idim) = unspecified_int4  ! bookkeeping
!	   ENDDO

* ... initialize memory diagnostics
	   current_cmnd_uvar = nu
	   deferred_committed_mem = 0

* ... evaluate the next sub-expression
	   CALL TRANSFER_CONTEXT( cx_cmnd, cx_expr )
	   CALL INTERP_STACK(  nu, cx_expr,
     .			      EXPR_NAME(nu), status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* ... check for legal data type
	   IF ( dtype .NE. ptype_native ) THEN
	      IF (mr_type(is_mr(isp)) .NE. dtype ) GOTO 5100
	   ENDIF

* ... add memory occupied by this result to the on-hold total of arguments
	   completed_args_mem = completed_args_mem + mr_size(is_mr(isp))

* tag the result for what it is
* note that since aux vars may now be returned on the ISP stack, isp may be GT nu
	   is_act(isp) = isact_var_ready

* Ticket 2278
* When defining an extra attribute based on the computation, e.g. dates that 
* went intothe modulo calculation. Has an attribute name and value have been 
* stored. natt_computed_var and the attribute name and string have been set 
* e.g. in the modulo regridding routine. 

	   IF (natt_computed_var .GT. 0) THEN

	      IF (STR_SAME (attstr_computed_var(1:10), 'do not set') .EQ. 0)  GOTO 290
     
* get the variable - will put an attrib onto it
	      cat = cx_category(cx_expr)
	      var = cx_variable(cx_expr)
	      varcode = VAR_CODE( cat, var )

* Ticket 2489: Skip this if it is an expression such as EX#2; There is 
* no varid to attach the attribute to.

	      IF (INDEX(varcode, '#') .GT. 0) GOTO 290

	      dset = cx_data_set( cx_expr )
	      varcode = SANITARY_VAR_CODE( cat, var )

	      IF (dset .EQ. pdset_irrelevant) dset = pdset_uvars
	      IF (cat .EQ. cat_user_var .AND. dset .LE. pdset_irrelevant)  dset = pdset_uvars
	      CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	      IF (status .NE. ferr_ok) THEN
	         IF (cat .EQ. cat_user_var .AND. dset.GE.pdset_irrelevant) THEN
	            dset = pdset_uvars
		    CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	         ENDIF
	      ENDIF

	      IF (status .EQ. ferr_ok) THEN
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname_computed_var, 
     .                attid, status) 
                 IF (attid .GT. 0) THEN
                    CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                attname_computed_var, attype, attlen, attoutflag, status )
                    attlen = TM_LENSTR1(attstr_computed_var)
                    CALL CD_REPLACE_ATTR (dset, varid, attname_computed_var, 
     .                attype, attlen, attstr_computed_var, dummy, status)
                 ELSE
                    attype = NCCHAR
                    attoutflag = 1
	            attlen = TM_LENSTR1(attstr_computed_var)
                    CALL CD_PUT_NEW_ATTR (dset, varid, attname_computed_var, 
     .                 attype, attlen, attoutflag, 
     .                 attstr_computed_var, 1., status) 
                    IF (status .NE. ferr_ok) status = ferr_ok  ! ignore
	         ENDIF
	      ENDIF

  290	      CONTINUE
	      CALL RESET_CLIMAT_ATT

	   ENDIF


! temp diagnostic
	   IF ( cx_stack_ptr .NE. cx_cmnd+nu ) CALL WARN( 'get_cmnd_???')
 300	CONTINUE

* turn off oversizing flag for next command
	cx_ovsiz(dflt_cx) = ovsiz_off
	RETURN

* error exit
 5000	RETURN
 5100	buff = VAR_CODE(mr_category(is_mr(nu)),mr_variable(is_mr(nu)))
* ... copied from INTERP_STACK error exit ... probably over-kill
	CALL RELEASE_DYN_WORK_SPACE
        CALL DELETE_MRS_IN_PROGRESS
        CALL UNPROTECT_ALL_MRS
        CALL DEALLO_ALL_GRIDS
        CALL CLEAN_STACK
	CALL ERRMSG( ferr_data_type, status,buff(:TM_LENSTR1(buff)), *5000 )
	END
