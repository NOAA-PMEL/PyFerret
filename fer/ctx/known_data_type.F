	SUBROUTINE KNOWN_DATA_TYPE ( cx )

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
* find the data type for the given variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V533:  7/01
* V554 1/04 *acm* - add counter_var 
*      *kms* 11/10 added cat_pystat_var
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V698 3/16 *sh* added logic for cat_ef_time2d (ACTS_LIKE_FVAR)
* V710 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
        include 'xpyvar_info.cmn'

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	LOGICAL ACTS_LIKE_FVAR
	INTEGER	dset, data_set, category, variable, idim, trans, itype

* initialize
	category = cx_category(cx)
	variable = cx_variable(cx)
	data_set = cx_data_set(cx)

* different procedure for determining grid from each category
	IF     ( ACTS_LIKE_FVAR(category) ) THEN

	   cx_type(cx) =  ds_var_type(variable)

	ELSEIF ( category .EQ. cat_pystat_var ) THEN

           cx_type(cx) =  pyvar_type(variable)

	ELSEIF ( category .EQ. cat_user_var ) THEN

* ... allow "abstract" expressions that don't depend on data set
	   IF ( .NOT.uvar_need_dset(variable)
     .	   .OR. data_set .EQ. unspecified_int4 ) THEN
	      dset = pdset_irrelevant
	   ELSE
	      dset = data_set
	   ENDIF
	   
	   CALL GET_SAVED_UVAR_DTYPE(variable, dset, itype)
	   cx_type(cx) = itype

	ELSEIF ( category .EQ. cat_temp_var ) THEN
!! here, variable may be up to 2000, mr_type is dimension 151
	   cx_type(cx) = mr_type( variable )  

	ELSEIF ( category .EQ. cat_counter_var ) THEN

	   cx_type(cx) = ptype_float

	ELSEIF ( category .EQ. cat_attrib_val ) THEN  !should have been set elsewhere

	   IF (cx_type(cx) .EQ. 0 .OR. 
     .         cx_type(cx) .EQ. unspecified_int4 ) 
     .        STOP 'unknown var.att in KNOWN_DATA_TYPE'

	ELSE

	   STOP 'KNOWN_TYP'

	ENDIF

* For string variables the result data type of @NGD and @NBD is float.  
	IF (cx_type(cx) .EQ. ptype_string) THEN
	   DO 100, idim = 1, nferdims
	      trans = cx_trans(idim, cx) 
	      IF (trans .EQ. trans_good_pt    .OR.
     .            trans .EQ. trans_bad_pt     .OR.
     .            trans .EQ. trans_4d_good_pt .OR.
     .            trans .EQ. trans_4d_bad_pt ) THEN
                  cx_type(cx) = ptype_float
       	       ENDIF
 100	   CONTINUE
        ENDIF

	RETURN
	END
