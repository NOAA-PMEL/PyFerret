	SUBROUTINE TM_DSET_USE_GRIDS( dset )

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
* bump the use counts on all grids used by variables in this data set

* Steve Hankin  NOAA/PMEL/TMAP

* V510 *sh* 4/00
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V7.4  6/18 *acm* Allow cleanup with grids created while initializing dsg files

* Include files
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'


* Argument definitions
	INTEGER dset

* Local definitions
	INTEGER i

* bump the use counts of grids used by variables in this data set
* if called as part of cleanup in unsuccessful grid creation, 
* ds_var_setnum(i) may have been set, but not ds_grid_number(i)

	DO 50 i = 1, maxvars
	   IF ( ds_var_setnum(i) .EQ. dset 
     .         .AND. ds_grid_number(i).NE.int4_init) THEN
	      CALL TM_USE_DYN_GRID( ds_grid_number(i))
	   ENDIF
 50	CONTINUE

* Do we need to bump up the use-count of the xlation grid?

c	IF (dsg_ragged(dset)) THEN
c	   IF (dsg_xlate_grid(dset) .NE. int4_init) 
c     .         CALL TM_USE_DYN_GRID( dsg_xlate_grid(dset) )
c	ENDIF 

* always successful completion
	RETURN

	END
