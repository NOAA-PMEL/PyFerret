	SUBROUTINE REPORT_DEPENDENCY
     .		( cx, isp_stk_lev,  uvar, item, work, cx_status, desist )

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
* display a line describing a single variable of a dependency tree
* with indentation to communicate tree structure

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V685+  *sh* 9/13
* V698  3/16 *sh* added logic for cat_ef_time2d (ACTS_LIKE_FVAR)

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'


* calling argument declarations:
	LOGICAL desist
	INTEGER cx, isp_stk_lev, uvar, item, cx_status, device
	INTEGER	work(dependency_max_recs, dependency_nfields)

* internal variable declarations:
	LOGICAL ACTS_LIKE_FVAR, initialize, tell_dset, its_aux
	INTEGER TM_LENSTR1, llen, cat, var, dset, blanks, 
     .		istart, iend, len2, dnamlen
	CHARACTER*1024 GET_UNIQUE_DSET_NAME, VAR_CODE, VAR_TITLE, line_buff,
     .		dsetname

	INTEGER tree_level

	PARAMETER (its_aux = .FALSE.)

* initialize
	tree_level = isp_stk_lev - dependency_root_level

! left as ifdef in case it is useful to turn this code on for future debugging
# ifdef debug_dependency_tree
	cat  = cx_category(cx)
	var  = cx_variable(cx)
	dset = cx_data_set(cx)
	IF (dset .EQ. unspecified_int4) dset = pdset_irrelevant
	line_buff = ' '
	blanks = 2 * (tree_level-1)
	initialize = tree_level .EQ. 1

* announce the default dataset at the outset
	IF (initialize .AND. dset .NE. pdset_irrelevant) THEN
	   dsetname = GET_UNIQUE_DSET_NAME( dset )
	   CALL SPLIT_LIST( pttmode_explct, show_lun, 
     .		'   in default dataset '//dsetname, 0)
	ENDIF

* prepare the dataset name if it needs to be documented
	tell_dset = dset .NE. pdset_irrelevant
     .	      .AND. dset .NE. dependency_dflt_dset
	IF ( tell_dset ) THEN
	   dsetname = '[d='
	   dsetname(4:) = GET_UNIQUE_DSET_NAME( dset )
	   dnamlen = TM_LENSTR1(dsetname)
	   dsetname(dnamlen:dnamlen) = ']'
	ENDIF

* document this variable
	IF ( cx_status .EQ. ferr_unknown_variable ) THEN
	   istart = uvar_item_start(item, uvar)
	   iend   = uvar_item_end  (item, uvar)
	   line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	   llen = (iend-istart+1) + blanks
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
	   line_buff(llen+1:) = '   (unknown variable)'
	ELSEIF ( cx_status .EQ. ferr_unknown_data_set ) THEN
	   istart = uvar_item_start(item, uvar)
	   iend   = uvar_item_end  (item, uvar)
	   line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	   llen = (iend-istart+1) + blanks
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
	   line_buff(llen+1:) = '   (unknown dataset)'
	ELSEIF ( cx_status .EQ. ferr_unknown_auxvar ) THEN
	   istart = uvar_item_start(item, uvar) + var - 1
	   iend   = uvar_item_start(item, uvar) + cat - 1
	   line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	   llen = (iend-istart+1) + blanks
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
	   line_buff(llen+1:) = '   (unknown auxiliary variable)'
	ELSEIF (cat .EQ. cat_user_var) THEN
!	   line_buff(blanks+1:) = FULL_UVAR_NAME(var, llen)
!	   llen = llen + blanks
	   line_buff(blanks+1:) = VAR_CODE(cat, var)
	   llen = TM_LENSTR1(line_buff)
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
	   line_buff(llen+1:) = ' = ' //uvar_text(var)
	ELSEIF ( ACTS_LIKE_FVAR(cat) ) THEN
	   line_buff(blanks+1:) = VAR_CODE(cat, var)
	   llen = TM_LENSTR1(line_buff)
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
        ELSEIF ( cat .EQ. cat_pystat_var ) THEN
           line_buff(blanks+1:) = VAR_CODE(cat, var)
           llen = TM_LENSTR1(line_buff)
           IF (tell_dset) THEN
              line_buff(llen+1:) = dsetname
              llen = llen + dnamlen
           ENDIF
           line_buff(llen+1:) = ' = ' //VAR_TITLE( cat )
	ELSE
* ... does this ever happen?
	   line_buff(blanks+1:) = VAR_CODE(cat, var)
	   llen = TM_LENSTR1(line_buff)
	   IF (tell_dset) THEN
	      line_buff(llen+1:) = dsetname
	      llen = llen + dnamlen
	   ENDIF
	   line_buff(llen+1:) = ' = ' //VAR_TITLE( cat )
	ENDIF

	CALL SPLIT_LIST( pttmode_explct, show_lun, 
     .		'   '//line_buff, 0)
	    
# endif

* save the item in the dependency tree
	CALL STORE_DEPENDENCY(its_aux, cx, tree_level,  uvar, item, cx_status,
     .			work(1,1), work(1,2), work(1,3), work(1,4), 
     .			work(1,5), work(1,6), work(1,7), work(1,8), desist )


	RETURN
	END
