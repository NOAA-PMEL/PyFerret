	SUBROUTINE GET_DEPENDENCY_STATUS
     .			(  expr, status_string, slen, status )

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
* set up the xdependency_tree COMMON area by running the dependency crawl
* return a line describing the outcome

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V685+  *sh* 11/13 
* V690  *sg* 5/14 - bug fix: changed declaraation type of "memory"
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'

* calling argument declarations:
	INTEGER status, slen
	CHARACTER*(*) expr, status_string

* internal variable declarations:
	INTEGER TM_LENSTR1, base_isp, base_cx

        base_isp = 8
        base_cx = 8
        CALL INIT_DEPENDENCIES( cx_last, .TRUE., base_isp, status )
        IF (status.NE. ferr_ok) GOTO 5000

        CALL CRAWL_DEPENDENCIES ( expr, base_cx, status)
        IF (status .NE. ferr_ok) GOTO 5000

	IF (    dependency_status .EQ. df_valid_var) THEN
	   status_string = 'AVAILABLE'
        ELSEIF (dependency_status .EQ. df_unknown_auxvar) THEN
	   status_string = 'UNKNOWN AUXILIARY VARIABLE'
	ELSEIF (dependency_status .EQ. df_unknown_var) THEN
	   status_string = 'UNKNOWN VARIABLE'
	ELSEIF (dependency_status .EQ. df_unknown_grid) THEN
	   status_string = 'UNKNOWN GRID'
	ELSEIF (dependency_status .EQ. df_syntax_error) THEN
	   status_string = 'ERROR IN EXPRESSION'
	ELSEIF (dependency_status .EQ. df_unknown_dataset) THEN
	   status_string = 'UNKNOWN DATASET'
	ELSEIF (dependency_status .EQ. df_recursion_error) THEN
	   status_string = 'ILLEGAL RECURSIVE VARIABLES'
	ELSE
	   STOP 'repl_exprns_depend_tree'
	ENDIF
	slen = TM_LENSTR1(status_string)

* if status incicates a problem, append a substring from the variable definitions
	IF (dependency_status .NE. df_valid_var ) THEN
	   status_string(slen+1:slen+2) = ': '
	   CALL DEPEND_STAT_VAR( workmem(plegacy_work_buffer)%ptr,
     .					status_string(slen+3:) )
	   slen = TM_LENSTR1(status_string)
	ENDIF

 5000	CALL RELEASE_DYN_WORK_SPACE
	RETURN
	END

