	SUBROUTINE FIND_MEM_VAR_INCL_C_CACHE( cx, mr, by_value, memory )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* try a traditional Ferret memory cache search.
*  If it fails, as a last ditch, look to see if the full range variable
* is available in c cache (normally used for CF auxiliary variables)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V695+ 5/15 *sh*
* V698  1/16 *sh* -- continue on to check cache whenever mr=0 from FIND_MEM_VAR

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER cx, mr
	LOGICAL by_value
	REAL    memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL valid_search
	INTEGER CGRID_SIZE, var, cat, npts, status
        REAL*8  c_cache_ptr

* first try a normal cache search
	CALL FIND_MEM_VAR( cx, mr, search_by_value, valid_search )
	IF (mr.NE.mr_nonexist) RETURN   ! 1/16 ignore valid_search

* initialize
	var = cx_variable(cx)
	cat = cx_category(cx)
* ... aggegated vars always map to file var of same cx_variable number
	IF (cat .EQ. cat_agg_f_var) cat = cat_file_var

* last ditch attempt -- is this variable available in c memory cache
* (used to speed recall of CF auxiliary variables)
* NOTE: error conditions are not properly trapped
	IF ( cat .EQ. cat_file_var ) THEN
	   c_cache_ptr = ds_var_c_cache_ptr(var)
	ELSEIF ( cat .EQ. cat_user_var ) THEN
	   c_cache_ptr =  uvar_c_cache_ptr(var) 
	ELSE
	   c_cache_ptr =  0.0
	ENDIF

	IF (c_cache_ptr .NE. 0.0) THEN
	   CALL CREATE_FULL_RANGE_CONTEXT ( cx, cx_buff, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	   CALL CREATE_MEM_VAR( cx_buff, mr, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	   npts = CGRID_SIZE(cx_buff)
	   CALL RECOVER_CACHED_FULL_ARRAY(memory(1,mr_blk1(mr)),
     .				npts, c_cache_ptr)
	   IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('c-found',mr,point_to_mr)
	   RETURN  ! YAHOO!  Made a last didch cache hit.
	ENDIF
	
* failure exit
 5000	mr = mr_nonexist
	RETURN
	END
