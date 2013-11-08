	SUBROUTINE DISPLAY_DEPENDENCY_SUB (
     .            next, level, flag, dataset, var_or_start, cat_or_end,
     .            parent_uv, uv_item, is_done )

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
* display the dependency tree structure that is stored in a memory work array

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V685+  *sh* 11/13 

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'


* calling argument declarations:
        INTEGER next         (dependency_max_recs),
     .          level        (dependency_max_recs),
     .          flag         (dependency_max_recs),
     .          dataset      (dependency_max_recs),
     .          var_or_start (dependency_max_recs),
     .          cat_or_end   (dependency_max_recs),
     .          parent_uv    (dependency_max_recs),
     .          uv_item      (dependency_max_recs)
	LOGICAL is_done      (dependency_max_recs)


* internal variable declarations:
	LOGICAL tell_dset, dup, state_dflt_dataset
	INTEGER	TM_LENSTR1,rec, ntrees, size, tree_size, tree_rec1,
     .		var, cat, tree_level, dset, uvar, item, ndone,
     .		blanks, llen, istart, iend, len2, dnamlen, tree, i
	CHARACTER*1024 GET_UNIQUE_DSET_NAME, VAR_CODE, VAR_TITLE, line_buff,
     .		dsetname

* initialize
	IF (  dependency_tree_show_fvars
     .	.AND. .NOT.dependency_tree_show_uvars ) GOTO 2000  ! show file var list only
* ... traverse forward linked list to count the number and set all as not yet displayed
	rec = 1
	ntrees = 0
 10	IF (next(rec) .GT. rec) THEN
	   ntrees = ntrees + 1
	   is_done(rec) = .FALSE.
	   rec = next(rec)
	   GOTO 10
	ENDIF
	ndone = 0

* do we need to announce the default dataset
	state_dflt_dataset = dependency_dflt_dset .NE. pdset_irrelevant

	DO 1000 tree = 1, ntrees

* find the largest tree not yet displayed
	   rec = 1
	   tree_size = 0
 20	   IF (next(rec) .GT. rec) THEN
	      size = next(rec) - rec
	      IF (.NOT.is_done(rec) .AND. (size.GT.tree_size)) THEN
	         tree_size = size
	         tree_rec1 = rec    ! starting record of the largest tree
	      ENDIF
	      rec = next(rec)
	      GOTO 20
	   ENDIF

* decide if this one has already been displayed
	   rec = 1
 30	   IF (next(rec) .GT. rec) THEN
	      IF (is_done(rec)) THEN
	         DO 50 i = rec, next(rec)-1
	            IF (flag(i) .NE. df_valid_var) GOTO 50
                    dup = flag        (i) .EQ. flag        (tree_rec1)
     .              .AND. dataset     (i) .EQ. dataset     (tree_rec1)
     .              .AND. var_or_start(i) .EQ. var_or_start(tree_rec1)
     .              .AND. cat_or_end  (i) .EQ. cat_or_end  (tree_rec1)
	            IF (dup) GOTO 900    ! already displayed as part of a previous tree
 50		 CONTINUE
	      ENDIF
	      rec = next(rec)
	      GOTO 30
	   ENDIF

* display one tree, record by record
	   DO 500 rec = tree_rec1, tree_rec1+tree_size-1

* initialize this record
	      tree_level = level(rec)
	      var  = var_or_start(rec)
	      cat  = cat_or_end  (rec)
	      dset = dataset(rec)
	      uvar = parent_uv(rec)
	      item = uv_item(rec)
	      line_buff = ' '
	      blanks = 2 * (tree_level-1)

* prepare the dataset name if it needs to be documented
	      tell_dset = dset .NE. pdset_irrelevant
     .	            .AND. dset .NE. dependency_dflt_dset
	      IF ( tell_dset ) THEN
	         dsetname = '[d='
	         dsetname(4:) = GET_UNIQUE_DSET_NAME( dset )
	         dnamlen = TM_LENSTR1(dsetname)
	         dsetname(dnamlen:dnamlen) = ']'
	      ENDIF

* document this variable
	      IF ( flag(rec) .EQ. df_unknown_var ) THEN
	         istart = uvar_item_start(item, uvar)
	         iend   = uvar_item_end  (item, uvar)
	         line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	         llen = (iend-istart+1) + blanks
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   (unknown variable)'
	      ELSEIF ( flag(rec) .EQ. df_unknown_dataset ) THEN
	         istart = uvar_item_start(item, uvar)
	         iend   = uvar_item_end  (item, uvar)
	         line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	         llen = (iend-istart+1) + blanks
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   (unknown dataset)'
	      ELSEIF ( flag(rec) .EQ. df_unknown_grid ) THEN
	         istart = uvar_item_start(item, uvar)
	         iend   = uvar_item_end  (item, uvar)
	         line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	         llen = (iend-istart+1) + blanks
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   (unknown grid)'
	      ELSEIF ( flag(rec) .EQ. df_syntax_error ) THEN
	         istart = uvar_item_start(item, uvar)
	         iend   = uvar_item_end  (item, uvar)
	         line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	         llen = (iend-istart+1) + blanks
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   *** WARNING: ERROR IN DEFINITION'
	      ELSEIF ( flag(rec) .EQ. df_unknown_auxvar ) THEN
	         istart = uvar_item_start(item, uvar) + var - 1
	         iend   = uvar_item_start(item, uvar) + cat - 1
	         line_buff(blanks+1:) = uvar_text(uvar)(istart:iend)
	         llen = (iend-istart+1) + blanks
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   (unknown auxiliary variable)'
	      ELSEIF (flag(rec) .EQ. df_defined_earlier) THEN
	         line_buff(blanks+1:) = VAR_CODE(cat, var)
	         llen = TM_LENSTR1(line_buff)
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = ' =  (defined above)'
	      ELSEIF (flag(rec) .EQ. df_recursion_error) THEN
	         line_buff(blanks+1:) = VAR_CODE(cat, var)
	         llen = TM_LENSTR1(line_buff)
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = ' =   *** WARNING: RECURSIVE DEFINITION.  See above'
	      ELSEIF (cat .EQ. cat_user_var) THEN
	         IF (.NOT.dependency_tree_show_uvars) GOTO 500
	         line_buff(blanks+1:) = VAR_CODE(cat, var)
	         llen = TM_LENSTR1(line_buff)
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = ' = ' //uvar_text(var)
	      ELSEIF (cat .EQ. cat_aggregate_var ) THEN
	         IF (.NOT. dependency_tree_show_uvars) GOTO 500
	         line_buff(blanks+1:) = VAR_CODE(cat, var)
	         llen = TM_LENSTR1(line_buff)
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
	         line_buff(llen+1:) = '   (aggregate variable)'
	      ELSEIF (cat .EQ. cat_file_var ) THEN
	         IF (.NOT.dependency_tree_show_fvars) GOTO 500
	         line_buff(blanks+1:) = VAR_CODE(cat, var)
	         llen = TM_LENSTR1(line_buff)
	         IF (tell_dset) THEN
	            line_buff(llen+1:) = dsetname
	            llen = llen + dnamlen
	         ENDIF
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
   
* name the default dataset just once
	IF (state_dflt_dataset) THEN
	   dsetname = GET_UNIQUE_DSET_NAME( dependency_dflt_dset )
	   CALL SPLIT_LIST( pttmode_explct, show_lun, 
     .		'   in default dataset '//dsetname, 0)
	   state_dflt_dataset = .FALSE.
	ENDIF

* display a line of the tree
	      CALL SPLIT_LIST( pttmode_explct, show_lun, 
     .	      	'   '//line_buff, 0)

 500	   CONTINUE


! signal that this tree has been displayed
 900	   is_done(tree_rec1) = .TRUE.

 1000	CONTINUE
	RETURN

**************************
* special display mode to show only a simple list of unique file variables
 2000	DO 2100 rec = 1, dependency_nrows

	   var  = var_or_start(rec)
	   cat  = cat_or_end  (rec)
	   dset = dataset(rec)
	   uvar = parent_uv(rec)
	   item = uv_item(rec)

* format one record
	   IF     ( flag(rec) .EQ. df_valid_var 
     .	 .AND.      cat       .EQ. cat_user_var       ) THEN
	      GOTO 2100					! skip user variables
	   ELSEIF ( flag(rec) .EQ. df_defined_earlier
     .	  .OR.      flag(rec) .EQ. df_recursion_error ) THEN
	      GOTO 2100					! skip irrelevant flags
	   ELSEIF ( flag(rec) .EQ. df_valid_var
     .	 .AND.      cat       .EQ. cat_file_var       ) THEN
* valid file variable
* ... if this record has already been displayed do not do so again
	      DO 2050 i = 1, rec-1
                 dup = flag        (i) .EQ. flag        (rec)
     .           .AND. dataset     (i) .EQ. dataset     (rec)
     .           .AND. var_or_start(i) .EQ. var_or_start(rec)
     .           .AND. cat_or_end  (i) .EQ. cat_or_end  (rec)
	         IF (dup) GOTO 2100
 2050	      CONTINUE
	      line_buff = VAR_CODE(cat,var)		! valid file variable
	      llen = TM_LENSTR1(line_buff)
	      dsetname = '[d='
	      dsetname(4:) = GET_UNIQUE_DSET_NAME( dset )
	      dnamlen = TM_LENSTR1(dsetname) + 1
	      dsetname(dnamlen:dnamlen) = ']'
	      line_buff = line_buff(:llen)//dsetname
	   ELSEIF ( flag(rec) .EQ. df_unknown_var ) THEN
	      istart = uvar_item_start(item, uvar)
	      iend   = uvar_item_end  (item, uvar)
	      line_buff = uvar_text(uvar)(istart:iend)//'  (unknown variable)'
	   ELSEIF ( flag(rec) .EQ. df_unknown_dataset ) THEN
	      istart = uvar_item_start(item, uvar)
	      iend   = uvar_item_end  (item, uvar)
	      line_buff = uvar_text(uvar)(istart:iend)//'  (unknown dataset)'
	   ELSEIF ( flag(rec) .EQ. df_unknown_grid ) THEN
	      istart = uvar_item_start(item, uvar)
	      iend   = uvar_item_end  (item, uvar)
	      line_buff = uvar_text(uvar)(istart:iend)//'  (unknown grid)'
	   ELSEIF ( flag(rec) .EQ. df_syntax_error ) THEN
	      istart = uvar_item_start(item, uvar)
	      iend   = uvar_item_end  (item, uvar)
	      line_buff = uvar_text(uvar)(istart:iend)//'   *** WARNING: SYNTAX ERROR'
	   ELSEIF ( flag(rec) .EQ. df_unknown_auxvar ) THEN
	      istart = uvar_item_start(item, uvar) + var - 1
	      iend   = uvar_item_start(item, uvar) + cat - 1
	      line_buff = uvar_text(uvar)(istart:iend)//'  (unknown auxiliary variable)'
	   ENDIF

	   CALL SPLIT_LIST( pttmode_explct, show_lun, '   '//line_buff, 0)


 2100	CONTINUE

	RETURN
	END
