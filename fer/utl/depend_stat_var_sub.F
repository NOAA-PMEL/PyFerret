	SUBROUTINE DEPEND_STAT_VAR_SUB ( string,
     .			next,         level,      flag,      dataset,
     .			var_or_start, cat_or_end, parent_uv, uv_item )

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
* get a string describing the variable that was the source of the dependency tree
* status flag

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V685+  *sh* 11/13 

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'

* calling argument declarations:
	INTEGER	next         (dependency_max_recs),
     .		level        (dependency_max_recs),
     .		flag         (dependency_max_recs),
     .		dataset      (dependency_max_recs),
     .		var_or_start (dependency_max_recs),
     .		cat_or_end   (dependency_max_recs),
     .		parent_uv    (dependency_max_recs),
     .		uv_item      (dependency_max_recs)
	CHARACTER(*) string

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
	INTEGER	row, uvar, istart, iend
	CHARACTER VAR_CODE*128

* initialize
	row    = dependency_status_rec
	uvar   = parent_uv(row)
	istart = var_or_start(row)
	iend   = cat_or_end(row)

* get the string showing the cause of the error
	IF (flag(row) .EQ. df_recursion_error) THEN
	   string = VAR_CODE(iend, istart)    ! (cat, var)
	ELSE
	   string = uvar_text(uvar)(istart:iend)
	ENDIF

	RETURN
	END
