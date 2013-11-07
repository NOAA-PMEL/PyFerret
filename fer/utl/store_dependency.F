	SUBROUTINE STORE_DEPENDENCY
     .		( cx, tree_lev,  uvar, item, cx_status,
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

* V685+  *sh* 10/13 

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'


* calling argument declarations:
	LOGICAL	desist
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
	LOGICAL	known_var, dup, in_target_defn, direct_parentage
	INTEGER	var, cat, dset, df_flag, row, current_lev, i,
     .		this_flag

* initialize
	cat    = cx_category(cx)
	var    = cx_variable(cx)
	dset   = cx_data_set(cx)
	IF (dset .EQ. unspecified_int4) dset = pdset_irrelevant
	desist = .FALSE.
	known_var = cx_status .NE. ferr_unknown_variable
     .	      .AND. cx_status .NE. ferr_unknown_auxvar

* increment pointer to the next item to be stored
	row = dependency_nrows + 1
	dependency_nrows = row

* store the things that are always valid
	next (row) = 0		! for clean housekeeping, not used
	level(row) = tree_lev
	dataset (row) = dset

* determine flag to save
	IF (     cx_status .EQ. ferr_unknown_variable ) THEN
	   this_flag = df_unknown_var
	ELSEIF ( cx_status .EQ. ferr_unknown_data_set ) THEN
	   this_flag = df_unknown_dataset
	ELSEIF ( cx_status .EQ. ferr_unknown_auxvar ) THEN
	   this_flag = df_unknown_auxvar
	ELSEIF ( cx_status .EQ. ferr_unknown_grid ) THEN
	   this_flag = df_unknown_grid
	ELSEIF ( cx_status .NE. ferr_ok ) THEN
	   this_flag = df_syntax_error
	ELSE
	   this_flag = df_valid_var
	ENDIF
	flag(row) = this_flag
	 
* store things differently depending upon whether variable is known
	IF ( cx_status .EQ. ferr_unknown_auxvar ) THEN
	   var_or_start(row) = uvar_item_start(item, uvar)
     .					  + var - 1
	   cat_or_end  (row) = uvar_item_start(item, uvar)
     .					  + cat - 1
	   parent_uv   (row) = uvar
	   uv_item     (row) = item
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
* 1. Where there are multiple references to any var in same definition as in
*    LET target = b + b 
* it is not necessary to report the variable b twice
* 2. Where there are multiple references to a user-defined vaiable in separate
* definitions as in
*    LET target = b + c
*    LET b = c
*    LET c = d
*    LET d = 1
* it is not necessry to report the dependencies of c twice.  The variable
* in_target_defn is used to track case 2 duplication (this) versus case 1
* 3. The existence of recursion is apparent through a duplicated defnition
*    that lies in the direct line of decent from another, as in this example
*    where "P" indicates a parent and "S" indicates a sibling variable
*    LET P0 = P1 + S1
*    LET P1 = P2 + S2
*    LET P2 = P3 + S3
* LET P3 = S1 would not create a recursion, but LET P3 = P1 would
* As we look backwards in the dependency tree the direct descendent parent is
* always the very first one at the next lower level.
	IF (  known_var
     .	.AND. row .GT. dependency_current_tree_start ) THEN
	   in_target_defn = .TRUE.  ! while we are inside the defn of target
	   current_lev = tree_lev
	   DO 100 i = row-1, dependency_current_tree_start, -1
	      IF ( level(i) .LT. current_lev ) THEN
	         in_target_defn = .FALSE.
	         direct_parentage = .TRUE.
	         current_lev = level(i)
	      ELSE
	         direct_parentage = .FALSE.  ! never true while in_target_defn
	      ENDIF
	      dup =   flag(i)         .EQ. flag(row)
     .          .AND. dataset(i)      .EQ. dataset(row)
     .		.AND. var_or_start(i) .EQ. var_or_start(row)
     .		.AND. cat_or_end  (i) .EQ. cat_or_end  (row)
	      IF (dup) THEN
* ... the row we are adding is already in the dependency tree
*     so desist in the analysis of this variable
	         desist = .TRUE.
	         IF (in_target_defn .AND. level(i).EQ.tree_lev) THEN
* ... case 1: twin sibling var need not be reported
	            row = row - 1
	            dependency_nrows = row
	         ELSEIF (direct_parentage) THEN
* ... case 3: a recursive definition has been found
	            flag(row) = df_recursion_error
	         ELSEIF (cat_or_end(i) .EQ. cat_user_var) THEN
* ... case 2: uvar used in separate vars needs only a cross-ref the second time
	            flag(row) = df_defined_earlier
	         ENDIF
	         GOTO 500
	      ENDIF
 100	   CONTINUE
	ENDIF

* maintain the forward link pointer
 500	next (dependency_current_tree_start) = row+1
	next (row+1) = row+1

* maintain global flag for status of tree
	IF (flag(row) .GT. dependency_status) THEN
	   dependency_status     = flag(row)
	   dependency_status_rec = row
	ENDIF

	RETURN
	END
