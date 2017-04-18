	SUBROUTINE CREATE_DYN_WRKSPC( rqst_size, ws, status )

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
* return the index of the workmem buffer used

* the hierarchy of calls is as follows
*   create_dyn_wrkspc ==> get_ws_dynmem.F ==> get_ws_mem.c ==> store_ws_ptr.F

* programmer - steve hankin
* V702 *sh*  1/17 trac enhancement ticket #2369 - dynamic memory

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include	'ferret.parm'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	ws, status
	INTEGER*8 rqst_size

* internal variable declarations:


* select the buffer to use
* ... we can get away with a simplistic solution, because there are so few
*     buffers to manage
	DO ws = 2, max_ws
	   IF (ws_size(ws) .EQ. 0) GOTO 10
	ENDDO
	CALL ERRMSG( ferr_internal, status, 'create_dyn_wrkspc', *5000)

* allocate the memory
 10	CALL GET_WS_DYNMEM( rqst_size, ws, status )

 5000	RETURN

	END

