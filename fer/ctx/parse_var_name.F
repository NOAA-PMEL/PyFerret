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

#ifdef unix
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	full_var_name
	INTEGER		var_cx, mods_cx, status

* internal variable decalrations:
	INTEGER		full_name_len, left_bracket, right_bracket, name_len,
     .			atsign, idim,  brac_nest, i
 	LOGICAL		name_only
* 1/97 addition
	INTEGER	num_it, it_start(max_item_list), it_end(max_item_list)


* separate simple name from possible modifiers
	full_name_len = LEN   ( full_var_name )
	left_bracket  = INDEX ( full_var_name, '[' )
	name_only     = left_bracket .EQ. 0
	IF ( name_only ) THEN
	   name_len = full_name_len
	ELSE
	   name_len = left_bracket - 1
* ... hunt for matching right bracket
	   brac_nest = 1
	   right_bracket = 0	! default to catch syntax errors
	   DO 200 i = left_bracket+1,full_name_len
	      IF ( full_var_name(i:i) .EQ. "[" ) THEN
	         brac_nest = brac_nest + 1
	      ELSEIF ( full_var_name(i:i) .EQ. "]" ) THEN
	         brac_nest = brac_nest - 1
	         IF ( brac_nest .EQ. 0 ) THEN
	            right_bracket = i
	            GOTO 210
	         ENDIF
	      ENDIF
 200	   CONTINUE
 210	   IF (   right_bracket .LE. left_bracket + 1
     .	     .OR. name_len      .LE. 0			) GOTO 5000
	ENDIF

* If full_var_name is b@asn, which can occur if b not yet loaded, return 
* name_len is just up to the @

        atsign = INDEX ( full_var_name, '@' )
        IF (atsign .GT. 0  .AND. name_only) THEN
           name_len = atsign - 1
    	   IF (   name_len      .LE. 0			) GOTO 5000
        ENDIF

* break up the "[...,...,...]" into comma-separated arguments
	IF ( .NOT.name_only ) THEN
	   CALL PARSE_COMMA_LIST (	full_var_name,
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
     .				full_var_name,
     .				mods_cx,
     .				max_item_list,
     .				num_it,
     .				it_start,
     .				it_end,
     .				unknown_qual_bad,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5900

* apply those context mods that don't need to know the grid
* apply them now, before looking for the variable name, because the data set is
* needed in the name search
	DO 100 idim = 1, 4
 100	cx_given(idim,var_cx) = .FALSE.			! *sh* 4/90 bug fix
	CALL APPLY_CONTEXT_MODS ( mods_cx, var_cx, nest_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5900

* identify the simple variable name
	CALL FIND_VAR_NAME( cx_data_set( var_cx ),
     .			    full_var_name(1:name_len),
     .			    cx_category( var_cx ),
     .			    cx_variable( var_cx ) )
	IF ( cx_variable( var_cx ) .EQ. munknown_var_name ) GOTO 5100

* set the dat atype of the variable
	IF ( cx_category(var_cx) .EQ. cat_file_var ) THEN
	   cx_type(var_cx) = ptype_float   ! TO BE GENERALIZED LATER (8/00)
	ELSEIF (cx_category(var_cx) .EQ. cat_calc_var ) THEN
	   cx_type(var_cx) = ptype_float
	ELSE
	   cx_type(var_cx) = unspecified_int4 ! inheritedt from definition
	ENDIF

* successful completion:
	RETURN

* error exits:
 5000	CALL ERRMSG
     .			  ( ferr_syntax, status, full_var_name, *5900 )
 5100	CALL ERRMSG
     .		( ferr_unknown_variable, status, full_var_name, *5900 )

 5900	RETURN
	END
