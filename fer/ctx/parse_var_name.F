	SUBROUTINE PARSE_VAR_NAME( full_var_name, var_cx, mods_cx, status )

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
* parse a full variable specification of the form:
*			(eg.) TEMP[X=170E:170W,K=+1D,T=188979]
* return the context modifications in the modification buffer passed
* and the updated (but not completed) variable context in the var context passed
* (var_cx must contain the full default context at the time of the call)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/27/86
* revision 0.1 - 2/25/87 - using PARAMETER munknown_var_name
* revision 0.2 - 3/23/87 - added new TMAP library: time axes, file var. names
*			   APPLY_CONTEXT_MODS now called in this routine
* revision 0.3 - 9/16/87 - unknown_qual_bad moved to FERRET.PARM
* revision 0.4 - 7/27/88 - added var_cx (default) to get_context_mods call
* V200:  4/25/90 - initialize cx_given (bug): it was passed along from default
*		   context if no mods were given.  Thus "list/i=101:105 temp"
*		   would have cx_given(x_dim)=.true. whereas
*		   "list/i=101:105 temp[x=@ave]" would have it .false.
* V402: 7/95 - Allow nested [] of the form vname1[G=vname2[D=dset]]
* V450: 1/97 - use local variables instead of xPROG_STATE vars for item_start
*		etc. to avoid conflict. (But share the max_item_list parameter
*		from the COMMON area.
* V530:  8/00 *sh* - set the variables data type
* V550: 10/02 *acm*- Look for @ in full_var_name when determining name_len
* V554 1/04 *acm* - add category cat_counter_var
* V600: 9/05 *acm* - add category cat_attrib_val
* V611 5/08 *acm* Fix bug 1578; dataset specifiers applied correctly var[d=1].attname
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 3/13 *sh* support for auxiliary regridding variables
*		 and SHOW VAR/DEPENDENCIES (unknown variable err not reported)
* V690 12/13 *sh* set auto-curvilinear actions that are needed
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
*       12/14 *sh* save bracket character positions for VTREE report
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V698  3/16 *sh* added logic for cat_ef_time2d (ACTS_LIKE_FVAR)
* v698 5/16  *acm* ticket 2413: New attribute keywords ncoordvars, coordnames
*                  return info about coordinate-variables in datasets
* V710 9/16  *acm* ticket 2465: Parsing expressions including attribute references.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xdependency_tree.cmn'
#include "netcdf.inc"

* calling argument declarations:
	CHARACTER*(*)	full_var_name
	INTEGER		var_cx, mods_cx, status

* internal variable declarations:
	LOGICAL         ACTS_LIKE_FVAR
	INTEGER		STR_SAME, TM_LENSTR1, 
     .                  full_name_len, left_bracket, right_bracket, 
     .			name_len, atsign, idim,  brac_nest, i,
     .                  dset, varid, attype, attlen, attoutflag,
     .                  attid, dot,
     .			curv_vars(nferdims), curv_cats(nferdims)
 	LOGICAL		name_only, do_err, aux_given
* 1/97 addition
	INTEGER	num_it, it_start(max_item_list), it_end(max_item_list)
        CHARACTER varname*512, attname*128, newname*512

        newname = full_var_name

* separate simple name from possible modifiers
	full_name_len = TM_LENSTR1 ( newname )
	left_bracket  = INDEX ( newname, '[' )
	name_only     = left_bracket .EQ. 0
	IF ( name_only ) THEN
	   name_len = full_name_len
	ELSE
	   name_len = left_bracket - 1
* ... hunt for matching right bracket
	   brac_nest = 1
	   right_bracket = 0	! default to catch syntax errors
	   DO i = left_bracket+1,full_name_len
	      IF ( newname(i:i) .EQ. "[" ) THEN
	         brac_nest = brac_nest + 1
	      ELSEIF ( newname(i:i) .EQ. "]" ) THEN
	         brac_nest = brac_nest - 1
	         IF ( brac_nest .EQ. 0 ) THEN
	            right_bracket = i
	            GOTO 210
	         ENDIF
	      ENDIF
	   ENDDO
 210	   IF (   right_bracket .LE. left_bracket + 1
     .	     .OR. name_len      .LE. 0			) GOTO 5000
	ENDIF

        IF (left_bracket .GT. 0) THEN
           dot = INDEX ( newname, '.' )
           IF (dot .GT. right_bracket) THEN  ! this is a var[d=].attribute
              newname = full_var_name(1:left_bracket-1)//
     .         full_var_name(left_bracket:right_bracket)//
     .         full_var_name(right_bracket+1:full_name_len)
              full_name_len = TM_LENSTR1(newname)
              name_len = full_name_len
              name_only = .TRUE.
           ENDIF
        ENDIF

* If full_var_name is b@asn, which can occur if b not yet loaded, return 
* name_len is just up to the @

        atsign = INDEX ( newname, '@' )
        IF (atsign .GT. 0  .AND. name_only) THEN
           name_len = atsign - 1
    	   IF (   name_len      .LE. 0			) GOTO 5000
        ENDIF

* break up the "[...,...,...]" into comma-separated arguments
	IF ( .NOT.name_only ) THEN
	   CALL PARSE_COMMA_LIST (	newname,
     .					left_bracket  + 1,
     .					right_bracket - 1,
     .					max_item_list,
     .					num_it,
     .					it_start,
     .					it_end,
     .					status		)
	   IF ( status .NE. ferr_ok ) GOTO 5900
	ELSE
	   num_it = 0
	ENDIF

*interpret the list of items - unknown items represent errors
	CALL GET_CONTEXT_MODS (	var_cx,
     .				newname,
     .				mods_cx,
     .				max_item_list,
     .				num_it,
     .				it_start,
     .				it_end,
     .				unknown_qual_bad,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5900

! no.  save exact position when error is detected in GET_CONTEXT_MODS
!* save the bracket position for VTREE unknown grid report
!	IF (cx_vtree_unknown_grid) THEN
!	   cx_vtree_var_or_start = left_bracket
!	   cx_vtree_cat_or_end   = right_bracket
!	ENDIF

* apply those context mods that don't need to know the grid
* apply them now, before looking for the variable name, because the data set is
* needed in the name search
	DO idim = 1, nferdims
	   cx_given(idim,var_cx) = .FALSE.			! *sh* 4/90 bug fix
 	ENDDO
	CALL APPLY_CONTEXT_MODS ( mods_cx, var_cx, nest_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5900

* identify the simple variable name
        CALL FIND_VAR_NAME( cx_data_set( var_cx ),
     .			    newname(1:name_len),
     .			    cx_category( var_cx ),
     .			    cx_variable( var_cx ) )

* always apply mods that describe the auxiliary regridding variables
	cx_naux(var_cx) = cx_naux(mods_cx)
	DO idim = 1, nferdims
	   cx_aux_arg_start(idim,var_cx) = cx_aux_arg_start(idim,mods_cx)
	   cx_aux_arg_end  (idim,var_cx) = cx_aux_arg_end  (idim,mods_cx)
	   cx_aux_stat     (idim,var_cx) = cx_aux_stat     (idim,mods_cx) 
	ENDDO

	IF ( cx_variable( var_cx ) .EQ. munknown_var_name ) THEN
* bail out if this variable cannot be identified
	   GOTO 5100
	ELSEIF (cx_category(var_cx) .EQ. cat_file_var
     .     .OR. cx_category(var_cx) .EQ. cat_user_var ) THEN
* fill in the default aux vars associated with this variable
	   CALL GET_AUTO_AUX_VARS( cx_variable(var_cx),
     .				   cx_category(var_cx),
     .				   cx_data_set(var_cx),
     .				   curv_vars, curv_cats )
	   DO idim = 1, nferdims
	      IF ( curv_vars(idim) .NE. unspecified_int4
     .	     .AND. cx_aux_stat(idim,var_cx) .EQ. paux_stat_na  ) THEN
	         cx_aux_cat (idim,var_cx) = curv_cats(idim)
	         cx_aux_var (idim,var_cx) = curv_vars(idim)
	         cx_aux_stat(idim,var_cx) = paux_stat_passed
	      ENDIF
	   ENDDO
	ENDIF

* set up automated curvilinear actions:  aux variable loads and regridding
* cases include
*  [z=pt, layerz], [z=lo:hi:delta,layerz], [z=lo:hi@ave,layerz], [gz=zax,layerz]
	cx_implicit_aux(var_cx) = .FALSE.   ! set default
	aux_given = .FALSE.
	DO idim = 1, nferdims
	   aux_given = aux_given .OR. cx_aux_given(idim,mods_cx)
	ENDDO
	IF ( aux_given ) THEN
	   IF ( cx_category(var_cx) .NE. cat_file_var
     .	  .AND. cx_category(var_cx) .NE. cat_user_var ) GOTO 5200
!	   IF (cx_mod_layerz .AND.
!     .	     .NOT. (cx_given(z_dim,mods_cx)) GOTO 5300
!	   IF (cx_mod_curvi .AND.
!     .	     .NOT.(cx_given(x_dim,mods_cx).AND.cx_given(y_dim,mods_cx))
!     .					 ) GOTO 5350
	   CALL APPLY_AUTO_CURV(newname(:full_name_len), var_cx, status)
	   IF (status .NE. ferr_ok) GOTO 5900
	ENDIF

* set the data type of the variable'
	IF ( ACTS_LIKE_FVAR(cx_category(var_cx)) ) THEN
	   cx_type(var_cx) = ptype_float   ! TO BE GENERALIZED LATER (8/00)
	ELSEIF ( cx_category(var_cx) .EQ. cat_pystat_var ) THEN
	   cx_type(var_cx) = ptype_float
	ELSEIF ( cx_category(var_cx) .EQ. cat_counter_var ) THEN
	   cx_type(var_cx) = ptype_float
	ELSEIF (cx_category(var_cx) .EQ. cat_attrib_val) THEN

           do_err = .TRUE.
           dset = cx_data_set(var_cx)
           IF (dset .EQ. unspecified_int4) dset = pdset_uvars  
           CALL  BREAK_VARATTNAME ( newname(1:name_len),
     .                              dset, varname, 
     .                              attname, varid, do_err, status )

           IF (status .NE. ferr_ok) GOTO 5100
           IF (varid .GE. 0) THEN
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,
     .               status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .           attid, attname, attype, attlen, attoutflag, status )

              IF (status .NE. ferr_ok) THEN
* Is it a pseudo-attribute for the variable?

                 IF (STR_SAME(attname, 'attnames') .EQ. 0 .OR. 
     .               STR_SAME(attname, 'dimnames') .EQ. 0 .OR. 
     .               STR_SAME(attname, 'varnames') .EQ. 0 .OR. 
     .               STR_SAME(attname, 'coordnames') .EQ. 0 ) THEN
                    cx_type(var_cx) = ptype_string
                 ELSEIF (STR_SAME(attname, 'ndims') .EQ. 0 .OR. 
     .                   STR_SAME(attname, 'type')  .EQ. 0 .OR. 
     .                   STR_SAME(attname, 'nvars') .EQ. 0 .OR. 
     .                   STR_SAME(attname, 'nattrs') .EQ. 0 .OR. 
     .                   STR_SAME(attname, 'ncoordvars') .EQ. 0) THEN
                    cx_type(var_cx) = ptype_float
                 ELSE
                    GOTO 5100
                 ENDIF
              ELSE

                 cx_type(var_cx) = ptype_float
                 IF (attype .EQ. NCCHAR) cx_type(var_cx) = ptype_string
              ENDIF

           ELSE

* Is it a pseudo-attribute at the dataset level?
              cx_type(var_cx) = ptype_float
              IF (STR_SAME (attname, 'varnames')   .EQ. 0 .OR.
     .            STR_SAME (attname, 'dimnames')   .EQ. 0 .OR.
     .            STR_SAME (attname, 'coordnames') .EQ. 0 .OR.
     .            STR_SAME (attname, 'attnames')   .EQ. 0 .OR.
     .            STR_SAME (attname, 'nvars')      .EQ. 0 .OR.
     .            STR_SAME (attname, 'nattrs')     .EQ. 0 .OR.
     .            STR_SAME (attname, 'ncoordvars') .EQ. 0 .OR. 
     .               STR_SAME(attname, 'coordnames') .EQ. 0 )
     .         cx_type(var_cx) = ptype_string
           ENDIF
	ELSE
	   cx_type(var_cx) = unspecified_int4 ! inherited from definition
	ENDIF

* successful completion:
        status = ferr_ok
	RETURN

* error exits:
 5000	CALL ERRMSG( ferr_syntax, status, full_var_name, *5900 )
 5100	IF ( dependency_tree_mode ) THEN
	   status = ferr_unknown_variable   ! set the status, but don't report
	   GOTO 5900
	ELSE
	   CALL ERRMSG
     .		( ferr_unknown_variable, status, full_var_name, *5900 )
	ENDIF
 5200	CALL ERRMSG( ferr_syntax, status,
     .		'LAYERZ and CURVI can only be used on LET variables'//
     .		' and file variables: '//full_var_name, *5900 )
! 5300	CALL ERRMSG( ferr_syntax, status,
!     .		'LAYERZ must be accompanied by explicit Z limits: '
!     .		//full_var_name, *5900 )
! 5350	CALL ERRMSG( ferr_syntax, status,
!     .		'CURVI must be accompanied by explicit X and Y limits: '
!     .		//full_var_name, *5900 )

 5900	RETURN
	END
