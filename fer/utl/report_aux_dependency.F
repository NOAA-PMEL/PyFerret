	SUBROUTINE REPORT_AUX_DEPENDENCY
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
* store a line describing a single variable of a dependency tree
* this routine is identical to REPORT_DEPENDENCY except that it passes a special
* flag to indicate that it is not a full-fledged dependency of definition.
* The key difference is that the parent of the aux variable is not being *defined*
* when this reference to the aux var occurs, so the check for recursive
* definitions should not look further up the tree

* for example, in
*let depth = z[g=temp]+ 0*temp[k=1]
*    LET tempz = temp[gz(depth)=zpts]
*    VTREE=all
*       TEMPZ = TEMP[GZ(DEPTH)=ZPTS]
*         TEMP
*           DEPTH = Z[G=TEMP]+ 0*TEMP[K=1]
*             TEMP
* The final line is an identical reference to TEMP repeated, but not a recursion


* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V690  *sh* 11/13 

        IMPLICIT NONE
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

* internal variable declarations
	INTEGER	tree_level
	LOGICAL its_aux

	PARAMETER (its_aux = .TRUE.)

* initialize
	tree_level = isp_stk_lev - dependency_root_level

* save the item in the dependency tree
	CALL STORE_DEPENDENCY(its_aux, cx, tree_level,  uvar, item, cx_status,
     .			work(1,1), work(1,2), work(1,3), work(1,4), 
     .			work(1,5), work(1,6), work(1,7), work(1,8), desist )


	RETURN
	END
