	SUBROUTINE INTERP_CONTEXT ( isp_base, cx, name_code, status )



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
* Determine the complete context of a user variable -- its grid
* and all region information including region info that is embedded inside
* of sub-definitions. For example 
*	LET expr = A[l=1:5]; LET A = SST[X=130E:80W]
* would need to include the X limits, too, in the final result 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510: 3/00 *sh* extracted from INTERP_STACK

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER		isp_base, cx, status
	CHARACTER*(*)	name_code

* internal variable declarations:
	LOGICAL		explicit_qualifiers
	INTEGER		UVAR_NUMBER, KNOWN_GRID, idim, uvar,
     .			start, end, mod_cx

* calling arguments:
* in	isp_base	- stack level at which to begin calculations
*			  ( all levels below remain untouched )
* in	cx		- context slot to return
* in	name_code	- name of uvar to be evaluated
* out	status		- completion code


* locate the requested variable in the uvar variables
	uvar = UVAR_NUMBER( name_code )
	IF ( uvar .EQ. unspecified_int4 ) CALL ERRMSG
     .		( ferr_internal, status, 'intp_stk', *5000 )
	
* initialize the default context for the quantity to evaluate
	cx_variable( cx ) = uvar
	cx_category( cx ) = cat_user_var
	cx_grid    ( cx ) =
     .		KNOWN_GRID(cx_data_set(cx),cat_user_var,uvar)

* initialize base level of the interpretation stack
* ... ( see PUSH_INTERP_STACK for isp>isp_base )
	isp = isp_base			! stack pointer
	is_cx  ( isp ) = cx	! context for stack activities
	is_uvar( isp ) = uvar	! pointer to text buffer

* when called the request is ALWAYS for an untransformed implicitly defined
* user variable (e.g. "LIST U" or "LIST SIN(X+Y)"
* determine its grid, if necessary - then flesh out the requested dimensions
	IF ( cx_grid(cx) .EQ. unspecified_int4 ) THEN
	   CALL GET_UVAR_GRID( uvar,cx_data_set(cx),status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   cx_grid(cx) =
     .		KNOWN_GRID(cx_data_set(cx),cat_user_var,uvar)
	ENDIF

* flesh out the requested dimensions (noting if there are any)
	explicit_qualifiers = .FALSE.
	DO 5 idim = 1, 4
	   IF ( uvar_given( idim, uvar ) .GE. uvlim_needed
     .	   .OR. cx_given( idim,cx ) ) THEN
	      explicit_qualifiers = explicit_qualifiers
     .			       .OR. cx_given(idim,cx )
	      CALL CONFINE_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL FLESH_OUT_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
 5	CONTINUE
	
* copy uvar_* limits into the output context
	DO 100 idim = 1, 4
	   IF ( cx_lo_ss(cx, idim) .EQ. unspecified_int4 ) THEN
	      IF ( uvar_given(idim,uvar) .GE. uvlim_gvn_gnl
     .       .AND. uvar_given(idim,uvar) .LE. uvlim_gvn_xact ) THEN
	         cx_lo_ss(cx, idim) = uvar_lo_ss(uvar, idim)
	         cx_hi_ss(cx, idim) = uvar_hi_ss(uvar, idim)
	         cx_lo_ww (idim, cx) = uvar_lo_ww(idim, uvar)
	         cx_hi_ww (idim, cx) = uvar_hi_ww(idim, uvar)
	         cx_by_ss (idim, cx) = uvar_by_ss(idim, uvar)
	      ENDIF
	   ENDIF
 100	CONTINUE

* if the expression is a single variable, then "EX#1" becmes that variable
	IF ( uvar_num_items(uvar) .EQ. 1
     . .AND. uvar_item_type(1,uvar) .EQ. alg_variable ) THEN
	   start = uvar_item_start( 1, uvar )
	   end   = uvar_item_end  ( 1, uvar )
	   CALL PARSE_NAM_DSET( uvar_text(uvar)(start:end),
     .				cx, cx_data_set(cx),
     .				cx_category(cx),
     .				cx_variable(cx), mod_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* fill in any unspecified limits with information from the grid
	CALL COMPLETE_MISSING_LIMITS(cx)

* success
	RETURN

* error exit - clean up
 5000	CALL RELEASE_WORK_SPC
	CALL DELETE_MRS_IN_PROGRESS
	CALL UNPROTECT_ALL_MRS
	CALL DEALLO_ALL_GRIDS
        CALL CLEAN_STACK
	RETURN
	END




