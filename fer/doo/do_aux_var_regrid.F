	SUBROUTINE DO_AUX_VAR_REGRID( memory, axis_list, naxis, 
     .				      src,  msrc,  src_cx,
     .				      dst,  mdst,  dst_cx,
     .				      aux1, maux1, aux1_cx,
     .				      aux2, maux2, aux2_cx,
     .				      aux3, maux3, aux3_cx,
     .				      status )

*
*
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
* perform a dispatch of regridding to the rountine appropriate
! for the dimensionailty

* v685 4/13 *sh*

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	naxis, axis_list(naxis), status,
     .		msrc, mdst, maux1, maux2, maux3,
     .		src_cx, dst_cx, aux1_cx, aux2_cx, aux3_cx
	REAL	memory( mem_blk_size, max_mem_blks )
        REAL    src(*), dst(*), aux1(*), aux2(*), aux3(*)


* local variable declarations
	INTEGER	wkblk

* initialize local variables
	status = ferr_ok

	IF ( naxis .EQ. 1) THEN
* Ansley:  note that this needs to be generalized for other axes than Z
* allocate a work array for a single profile of Z depths
	   CALL GET_WORK_SPC( m1hiz-m1loz+1, wkblk, status )
           IF ( status .NE. ferr_ok ) GOTO 5000

* do the 1D regridding
	   CALL DO_AUX_VAR_REGRID_1D( axis_list(1), 
     .				      src,  msrc,  src_cx,
     .				      dst,  mdst,  dst_cx,
     .				      aux1, maux1, aux1_cx,
     .				      memory(1,wkblk) )
	ELSE
*  ==> insert horizontal curvilinear regridding (naxis.EQ.2) here
	   STOP 'do_aux_var_regrid'
	ENDIF

* success
	CALL RELEASE_WORK_SPC
	RETURN

* error
 5000	RETURN

	END
