	SUBROUTINE GET_DYN_WORK_SPACE( i8_rqst_size, status )

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
* a convenience function to get the first slot in the work space table
* replicates the legacy work space management previous to dynamic F90 memory

* the hierarchy of calls is as follows
*   get_dyn_work_space ==> get_ws_dynmem.F ==> get_ws_mem.c ==> store_ws_ptr.F

* programmer - steve hankin
* V702 *sh*  1/17 trac enhancement ticket #2369 - dynamic memory

        IMPLICIT NONE

        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include	'ferret.parm'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER status
	INTEGER*8 i8_rqst_size


* slot 1 of working storage is managed specially
* through convenience routines GET_DYN_WORK_SPACE and RELEASE_DYN_WORK_SPACE

* If the work space that is currently allocated is already large enough, then
* we will simply let it be re-used
	IF (ws_size(plegacy_work_buffer) .GE. i8_rqst_size) THEN
	   status = ferr_ok
	   RETURN
	ELSEIF (ws_size(plegacy_work_buffer) .NE. 0) THEN
	   CALL RELEASE_DYN_WORK_SPACE
	ENDIF

* get the requested memory
	CALL GET_WS_DYNMEM(i8_rqst_size, plegacy_work_buffer, status)

	END

