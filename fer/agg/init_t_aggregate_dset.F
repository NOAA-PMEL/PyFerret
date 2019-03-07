	SUBROUTINE INIT_T_AGGREGATE_DSET(mem_filename_list,
     .			nagfiles, have_expr,
     .			dset_name, dset_path, dset_title,
     .			dset_num, tline, tmatch, toverlap, status)
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
* Assemble the needed arguments and call DEFINE_T_AGG_MC_DSET
* The time axis, if not provided, is generated from the time axis metadata
* read from each file to be aggregated.

* the incoming command syntax may be
*    DEFINE DATA/AGGREGATE/<dim>  agname = file1, file2, file3, ...
*  or
*    DEFINE DATA/AGGREGATE/<dim>  agname = <string_expression>


* admittedly, the hierarchy of calling names is not self-evident.  They are:
* top level: INIT_T_AGGREGATE_DSET
*            ==> DEFINE_T_AGG_MC_DSET
*                ==> BUILD_T_AGG_AXIS
*                ==> INIT_T_AGG_MC_DSET

* V695+ *sh* 11/15 new routine
* V698 *sh* 1/16  pass explicit path name (for "USE file.agg" compatibility)
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

* arguments:
*	mem_filename_list - if have_expr, the variable of names (input)
*	nagfiles	- number of files in the aggregation (input)
*       have_expr	- flag if filenames were listed or in an expr (input)
*	dset_name	- dataset name to assign to the aggregation (input)
*	dset_path	- path name to assign to the aggregation (input)
*	dset_title	- dataset title (input)
*       tline           - user-provided time axis to use (input, can be unspecified)
*       tmatch          - method of matching times to tline (input, can be empty);
*                         eg, '@ASN', '@BIN', '@XACT', '@XACT:0.001'
*       toverlap        - allowed number of overlapping time values per datset (input);
*                         negative (such as unspecified_int4) == unlimited
*	dset_num	- number assigned to this data set (output)
*	status		- return status (output)
*

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
        include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'   ! for is_mr

* Argument definitions
	LOGICAL		have_expr
	INTEGER		nagfiles, dset_num, tline, toverlap, status
	CHARACTER*(*)	dset_name, dset_path, dset_title, tmatch
	REAL		mem_filename_list(*)

* Internal declarations
	INTEGER	iset, slen, max_str_len
	CHARACTER*512 GET_STRING_ELEMENT_1D_X
        CHARACTER*512 memb_filename(nagfiles)

* initialize
	status = ferr_ok
	max_str_len = 512

* set up array of filenames
	IF (have_expr) THEN
*  ... filenames passed through a string variable
	  DO iset = 1, nagfiles
            memb_filename(iset) = GET_STRING_ELEMENT_1D_X(nagfiles,
     .                   mem_filename_list, iset, max_str_len, slen)
	  ENDDO
	ELSE
*  ... filenames given by user on command line
	  DO iset = 1, nagfiles
	    memb_filename(iset) = cmnd_buff(item_start(iset):item_end(iset))
	  ENDDO
	ENDIF

* now call the "real" routine
        CALL DEFINE_T_AGG_MC_DSET(dset_name, dset_path, dset_title,
     .            nagfiles, memb_filename, tline, tmatch, toverlap, 
     .            dset_num, status)

* done
	RETURN
	END
