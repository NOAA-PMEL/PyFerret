	CHARACTER*(*) FUNCTION VAR_TITLE_ONLY( cx, tlen )

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
* For ticket 2331, variable-title option
* create a title for a memory variable including only the variable's title
* If no title, then the variable name, or if an expression, its definition.

* programmer - Ansley Manke from VAR_TITLE
* revision 0.0 - 2/6/2017


        include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
        include 'errmsg.parm'

* calling argument declarations:
	INTEGER	cx, tlen

* internal variable declarations:
	LOGICAL 	NC_GET_ATTRIB, ACTS_LIKE_FVAR, got_it
	INTEGER 	TM_LENSTR1, maxlen, cat, var, idim, slen, 
     .			trans,uvar, item, start, end, dset, 
     .                  varid,attlen, attoutflag, status
        REAL            vals
	CHARACTER	VAR_CODE*128, SANITARY_VAR_CODE*128, PAREN_SURROUND*200,
     .                  VAR_UNITS*64, REPLACE_DEQ*180,   ! .GT. uvar_text_len
     .                  buff*2048, varname*128

* initialize
	maxlen	= LEN( VAR_TITLE_ONLY )

* initialize
	var = cx_variable( cx )
	cat = cx_category( cx )

	IF     ( ACTS_LIKE_FVAR(cat) ) THEN
	   IF ( ds_var_title( var )(1:1) .NE. ' ' ) THEN
	      VAR_TITLE_ONLY = ds_var_title( var )
	   ELSE
              dset = cx_data_set(cx)
              IF (dset .NE. pdset_irrelevant .AND. 
     .            dset .NE. unspecified_int4) THEN
! is it in the linked-list for dset. If so get attr info, including original 
                 varname = SANITARY_VAR_CODE( cat, var )
                 CALL CD_GET_VAR_ID (dset, varname, varid, status)
                 got_it = status .EQ. ferr_ok .AND.
     .              NC_GET_ATTRIB (dset, varid, 'long_name',
     .             .TRUE., varname, 2048, attlen, attoutflag,
     .             buff, vals)
                 VAR_TITLE_ONLY = buff
                 IF (VAR_TITLE_ONLY .EQ. ' ') VAR_TITLE_ONLY = varname
              ELSE
                 VAR_TITLE_ONLY = VAR_CODE( cat, var )
!	         VAR_TITLE_ONLY = '"'//var_title(:TM_LENSTR1(var_title))//'"'
              ENDIF
	   ENDIF

	ELSEIF ( cat .EQ. cat_user_var  ) THEN

	   IF ( uvar_title( var ) .EQ. ' ' ) THEN
	      VAR_TITLE_ONLY = uvar_name_code(var)
	      IF (VAR_TITLE_ONLY(1:3) .EQ. 'EX#') VAR_TITLE_ONLY = 
     .          REPLACE_DEQ(uvar_text(var)(:TM_LENSTR1(uvar_text(var)) ))
	   ELSE
	      VAR_TITLE_ONLY = uvar_title( var )
	   ENDIF


	ELSEIF ( cat .EQ. cat_attrib_val  ) THEN

	   uvar = cx_variable(cx)
	   VAR_TITLE_ONLY = uvar_text(uvar)

	   IF ( uvar_title( uvar ) .EQ. ' ' ) THEN
	      VAR_TITLE_ONLY = REPLACE_DEQ(
     .             uvar_text(uvar)(:TM_LENSTR1(uvar_text(uvar)) ))
	   ENDIF

	ELSEIF ( cat .EQ. cat_pseudo_var ) THEN

	   VAR_TITLE_ONLY = alg_pvar(var)

	ELSEIF ( cat .EQ. cat_dummy_var ) THEN

	   VAR_TITLE_ONLY = 'dummy'

	ELSEIF ( cat .EQ. cat_temp_var ) THEN

	   VAR_TITLE_ONLY = 'temp var'

	ELSEIF ( cat .EQ. cat_constant ) THEN

	   VAR_TITLE_ONLY = 'constant'

	ELSEIF ( cat .EQ. cat_const_var ) THEN

	   uvar = cx_variable(cx)/1000
           item = cx_variable(cx) - 1000*uvar
           start = uvar_item_start( item, uvar )
           end   = uvar_item_end  ( item, uvar )
	   VAR_TITLE_ONLY = uvar_text(uvar)(start:end)

	ELSEIF ( cat .EQ. cat_string ) THEN

* determine the string to be stored
           uvar = cx_variable(cx)/1000
           item = cx_variable(cx) - 1000*uvar
           start = uvar_item_start( item, uvar )
           end   = uvar_item_end  ( item, uvar )
	   VAR_TITLE_ONLY = uvar_text(uvar)(start:end)

	ELSEIF ( cat .EQ. cat_counter_var ) THEN

	   VAR_TITLE_ONLY = 'counter'

	ELSE

	   VAR_TITLE_ONLY = 'bad_cat'

	ENDIF

* if it wont fit indicate that with a "*" at the end
	tlen = MIN( maxlen, TM_LENSTR1(VAR_TITLE_ONLY) )
	IF ( tlen .EQ. maxlen ) VAR_TITLE_ONLY(maxlen:maxlen) = '*'

	RETURN
	END
