	SUBROUTINE STORE_DEPENDENCY
     .		( its_aux, cx, tree_lev,  uvar, item, cx_status,
     .		  next, level, flag, dataset, var_or_start, cat_or_end,
     .		  parent_uv, uv_item, desist )

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
* store a line describing a single variable of a dependency tree
* and send feedback if a duplication or a recursion has been detected that
* indicates further analysis of this tree level should desist

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V685  *sh* 10/13 
* V690  *sh* 11/13 - fixed problem in aux regridding variable dependencies
*                    by adding a new calling arg
* V693+ *sh* 1/21/15 - store regridding dependency lines
* V720  acm* 6/2017  Fix for ticket 2533, reporting ok for regridding to 
*                    undefined variable

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'  ! for vms_str_success
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'


* calling argument declarations:
	LOGICAL	its_aux, desist
	INTEGER cx, tree_lev, uvar, item, cx_status
	INTEGER	next         (dependency_max_recs),
     .		level        (dependency_max_recs),
     .		flag         (dependency_max_recs),
     .		dataset         (dependency_max_recs),
     .		var_or_start (dependency_max_recs),
     .		cat_or_end   (dependency_max_recs),
     .		parent_uv    (dependency_max_recs),
     .		uv_item      (dependency_max_recs)

*	next         - forward link connecting starts of trees 
*	level        - the child generation level in the tree
*	flag         - special conditions (see xdependency_tree parameters
*	var_or_start - variable number if known variable
*                      or start character of uvar item if unknown
*	cat_or_end   - category number if known variable
*                      or end   character of uvar item if unknown
*	parent_uv    - uvar number of parent variable (used when unknown var)
*	uv_item      - uvar item number within parent (used when unknown var)


* internal variable declarations:
	LOGICAL	known_var, dup, in_target_defn, direct_parentage,
     .		aux_broken_chain
	INTEGER	var, cat, dset, df_flag, row, current_lev, i,
     .		this_flag, nrg
	INTEGER STR_CASE_BLIND_COMPARE, vax_code

* initialize
	cat    = cx_category(cx)
	var    = cx_variable(cx)
	dset   = cx_data_set(cx)
	desist = .FALSE.

* Fix for ticket 2533, if known_var when bad status, and pcx_vtree_regrid_report,
* regridding to an undefined variable was reported ready

	known_var = cx_status .EQ. ferr_ok
cc     .         .OR. cx_status .EQ. pcx_vtree_regrid_report 
!     .	      .OR.  cx_status .EQ. ferr_unknown_auxvar  ! ???

* set up regrid target reports
	IF ( cx_status .EQ. pcx_vtree_regrid_report ) THEN
	  IF (cx_vtree_regrid_ntarget .EQ. 0 ) RETURN  ! ?for SST[G=SST] case?
	  nrg = 1
	ENDIF

* increment pointer to the next item to be stored
 10	row = dependency_nrows + 1
	dependency_nrows = row

* determine flag to save
	IF ( cx_status .EQ. pcx_vtree_regrid_report ) THEN
* ... special treatment for regridding report (12/14)
*     cx_vtree_unknown_grid tells if this is an unknown target error report
*     if its an error then cx_vtree_var_or_start / _cat_or_end hold "[]" posn
*     if not an error then cx_vtree_var_or_start / _cat_or_end hold
*         var & cat        if the regrid points to a target variable
*     or  unspecified_int4 if the regrid points to a named grid or axis
*     we do not report named grid or axis regridding.  Not a dependency.
*     Multiple target regrid variables get recorded in cx_vtree_regrid_ntarget
	   dset = cx_vtree_regrid_dset(nrg)
	   IF ( cx_vtree_unknown_grid(nrg)      ) THEN
	      this_flag = df_unknown_grid
	   ELSE
	      this_flag = df_valid_var
	   ENDIF
	ELSEIF ( cx_status .EQ. ferr_unknown_variable ) THEN
	   this_flag = df_unknown_var
	ELSEIF ( cx_status .EQ. ferr_unknown_data_set ) THEN
	   this_flag = df_unknown_dataset
	ELSEIF ( cx_status .EQ. ferr_unknown_auxvar ) THEN
	   this_flag = df_unknown_auxvar
	ELSEIF ( cx_status .EQ. ferr_unknown_grid ) THEN
	   this_flag = df_unknown_grid
	ELSEIF ( cx_status .NE. ferr_ok ) THEN
	   this_flag = df_syntax_error
	ELSEIF ( its_aux ) THEN
	   this_flag = df_valid_aux_var
	ELSE
	   this_flag = df_valid_var
	ENDIF
	 
* store the things that are always valid
	IF (dset .EQ. unspecified_int4) dset = pdset_irrelevant
	next (row) = 0		! for clean housekeeping, not used
	level(row) = tree_lev
	dataset (row) = dset
	flag(row) = this_flag

* store things differently depending upon whether variable is known
	IF ( cx_status .EQ. ferr_unknown_auxvar ) THEN
	   var_or_start(row) = uvar_item_start(item, uvar)
     .					  + var - 1
	   cat_or_end  (row) = uvar_item_start(item, uvar)
     .					  + cat - 1
	   parent_uv   (row) = uvar
	   uv_item     (row) = item
	ELSEIF ( cx_status .EQ. pcx_vtree_regrid_report ) THEN
	   var_or_start(row) = cx_vtree_var_or_start(nrg)
	   cat_or_end  (row) = cx_vtree_cat_or_end(nrg)
	   parent_uv   (row) = uvar
	   uv_item     (row) = pcx_vtree_regrid_report
	ELSEIF ( cx_status .NE. ferr_ok ) THEN
	   var_or_start(row) = uvar_item_start(item, uvar)
	   cat_or_end  (row) = uvar_item_end  (item, uvar)
	   parent_uv   (row) = uvar
	   uv_item     (row) = item
	ELSE
	   var_or_start(row) = var
	   cat_or_end  (row) = cat
	   parent_uv   (row) = unspecified_int4
	   uv_item     (row) = unspecified_int4
	ENDIF

* check for duplications:
* 1. in-line duplication
*  Where there are multiple references to any var in same definition as in
*    LET target = b + b 
*  it is not necessary to report the variable b twice
* 2. back-reference duplication
*  Where there are multiple references to a user-defined vaiable in separate
*  definitions as in
*    LET target = b + c
*    LET b = c
*    LET c = d
*    LET d = 1
*  it is not necessry to report the dependencies of c twice.  The variable
*  in_target_defn is used to track case 2 duplication (this) versus case 1
* 3. recursion
*  The existence of recursion is apparent through a duplicated defnition
*    that lies in the direct line of decent from another, as in this example
*    where "P" indicates a parent and "S" indicates a sibling variable
*    LET P0 = P1 + S1
*    LET P1 = P2 + S2
*    LET P2 = P3 + S3
*  LET P3 = S1 would not create a recursion, but LET P3 = P1 would
*  As we look backwards in the dependency tree the direct descendent parent is
*  always the very first one at the next lower level.
	IF (  known_var
     .	.AND. row .GT. dependency_current_tree_start ) THEN

* look for case 1:  twin sibling var that need not be reported
	   DO 100 i = row-1, dependency_current_tree_start, -1
	      IF     ( level(i) .GT. tree_lev ) THEN
	         GOTO 100                 ! inside of a child variable - ignore
	      ELSEIF ( level(i) .LT. tree_lev ) THEN
	         GOTO 110		  ! gone beyond start of this variable
	      ENDIF
* ...  flag cannot be used in the dup test because it gets reset for df_defined_earlier
*      since parent_uv is set to unspecified_int4 only for df_valid_var it serves
*      as a signal that the flag had been valid before it was reset to defined_earlier
	      IF ( cx_status .EQ. pcx_vtree_regrid_report 
     .	     .AND. flag(row) .EQ. df_unknown_grid         ) THEN
* ... ugly case of duplicate unknown grid target: "myvar[gx=mybad,gy=mybad]"
	        dup =   parent_uv(i)    .EQ. parent_uv(row)
     .		  .AND. flag(i)         .EQ. flag(row) 
	        IF (dup) THEN
	           vax_code = STR_CASE_BLIND_COMPARE(
     .			uvar_text(uvar)(var_or_start( i ):cat_or_end( i )),
     .			uvar_text(uvar)(var_or_start(row):cat_or_end(row)) )
	           dup = vax_code .EQ. vms_str_success
	        ENDIF
	      ELSE
	        dup =   parent_uv(i)    .EQ. parent_uv(row)
     .            .AND. dataset(i)      .EQ. dataset(row)
     .		  .AND. var_or_start(i) .EQ. var_or_start(row)
     .		  .AND. cat_or_end  (i) .EQ. cat_or_end  (row)
	      ENDIF
	      IF (dup) THEN
* ... found a twin -- do not report it
	         desist = .TRUE.
	         row = row - 1
	         dependency_nrows = row
	         GOTO 500
	      ENDIF
 100	   CONTINUE

* look for cases 1 and 3
 110	   in_target_defn = .NOT.its_aux    ! TRUE while we are inside the defn of target
	   aux_broken_chain = .FALSE.
	   current_lev = tree_lev
	   DO 200 i = row-1, dependency_current_tree_start, -1
	      IF ( flag(i) .EQ. df_valid_aux_var ) THEN
	         aux_broken_chain = .TRUE.    ! the chain of dependency stops at an aux var dependency
	         direct_parentage = .FALSE.
	      ENDIF
	      IF ( level(i) .LT. current_lev ) THEN
	         in_target_defn = .FALSE.
! doesnt this need to break the direct_parentage line permanently for all above?
! finding recursion through an aux var connection is still a problem, even if the parent is a uservar
! XXXXXXXXXXX   FIX PROBLEM:  FALSE RECURSIONS FROM AUX VAR DIRECT PARENTAGE
	         IF (.NOT.aux_broken_chain) direct_parentage = .TRUE.
	         current_lev = level(i)
	      ELSE
	         direct_parentage = .FALSE.  ! never true while in_target_defn
	      ENDIF
* ...  flag cannot be used in the dup test because it gets reset for df_defined_earlier
*      since parent_uv is set to unspecified_int4 only for df_valid_var it serves
*      as a signal that the flag had been valid before it was reset to defined_earlier
	      dup =   parent_uv(i)    .EQ. parent_uv(row)
     .          .AND. dataset(i)      .EQ. dataset(row)
     .		.AND. var_or_start(i) .EQ. var_or_start(row)
     .		.AND. cat_or_end  (i) .EQ. cat_or_end  (row)
	      IF (dup) THEN
* ... the row we are adding is already in the dependency tree
*     so desist in the analysis of this variable
	         IF (in_target_defn .AND. level(i).EQ.tree_lev) THEN
* ... case 1: twin sibling var need not be reported
*	SHOULD NEVER ARRIVE HERE
	            CALL WARN("crptn_store_dep")
	         ELSEIF (direct_parentage) THEN
* ... case 3: a recursive definition has been found
	            flag(row) = df_recursion_error
	            desist = .TRUE.
	            GOTO 500
	         ELSEIF (cat_or_end(i) .EQ. cat_user_var) THEN
* ... case 2: uvar used in separate vars needs only a cross-ref the second time
	            flag(row) = df_defined_earlier
	            desist = .TRUE.
	            GOTO 500
	         ENDIF
	      ENDIF
 200	   CONTINUE

	ENDIF

* maintain the forward link pointer
 500	next (dependency_current_tree_start) = row+1
	next (row+1) = row+1

* maintain global flag for status of tree
	IF (flag(row) .GT. dependency_status) THEN
	   dependency_status     = flag(row)
	   dependency_status_rec = row
	ENDIF

* if more regridding dependencies to report, loop back
	IF  ( cx_status .EQ. pcx_vtree_regrid_report 
     .	.AND. nrg .LT. cx_vtree_regrid_ntarget        ) THEN
	   nrg = nrg + 1
	   GOTO 10
	ENDIF

	RETURN
	END
