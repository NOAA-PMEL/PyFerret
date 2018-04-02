	SUBROUTINE GET_WS_DYNMEM( i8_rqst_size, ws, status )

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
* allocate the requested amount of dynamic memory (using c or python malloc)
* malloc is done in c (or python), but has to call back into F90 to set pointer
* the hierarchy of calls is as follows
*   get_ws_dynmem.F ==> get_ws_mem.c ==> store_ws_ptr.F

* programmer - steve hankin
* V702 *sh*  1/17 trac enhancement ticket #2369 - dynamic memory
* v74   *acm* 3/18 Introduce parameter abstractax_len replacing 99999999

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER	ws, status
	INTEGER*8 i8_rqst_size

* internal variable declarations:
	INTEGER	mr_del, num_slen
	CHARACTER  LEFINT8*20, num_str*20

* Check for unspecified limits on abstract axis
	IF ( i8_rqst_size .EQ. abstractax_len ) GOTO 5100

* allocate the dynamic memory
 200	CALL GET_WS_MEM(ws, i8_rqst_size, status)
        IF (status .NE. ferr_ok) GOTO 5300

* bookkeeping
        ws_size(ws) = i8_rqst_size
	RETURN

***********************
* error exits
 5000	RETURN
 5100	num_str = LEFINT8(i8_rqst_size,num_slen)
	CALL ERRMSG( ferr_dim_underspec, status,
     .    num_str(:num_slen)//' words were requested. ', *5110)
 5110	CALL WARN('Check for unspecified limits on an ABSTRACT axis')
	CALL WARN('Use the SHOW GRID command to see the axes')
	GOTO 5000

 5300	num_str = LEFINT8(i8_rqst_size,num_slen)
	CALL ERRMSG( ferr_insuff_memory,status,
     .	  num_str(:num_slen)//' words were requested. ', *5310)
 5310	CALL WARN('The system is refusing to supply this much memory')
        GOTO 5000

	END

