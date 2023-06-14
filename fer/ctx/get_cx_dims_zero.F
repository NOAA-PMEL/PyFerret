	SUBROUTINE GET_CX_DIMS_ZERO( cx, ndim, dim )

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
* determine the "shape" of the region specified in context cx by returning
* array, dim, which contains a permutation of the 4 defining axes 1,2,3,4
* and, ndim, the number of axes for which there is a range of data.
* The permutation is first those axes which contain a range of data
* (subsampled from the ordering 1,2,3,4) followed by the axes for which the
* context specifies only a single point

* e.g. for the expression SST[X=130e:70w,t="1-jan-1982":"31-dec-1983",Y=0]
* ndim = 2
* dim = 1,4,2,3		(the Z axis is normal for SST)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  5/18/89 - complete re-write from GET_CONTEXT_PLANE
* 10/2004 *acm*
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* For disp_init and extract_line, distinguish between axes which really 
* have one point, and those for which the hi and lo subscripts on the axes 
* are undefined. When undefined return zero from CX_DIM_LEN_ZERO. These axes 
* are listed last in the heirarchy in array dim().  This means we can 
* plot a variable with just one point in their range (if /HLIMITS or 
* /VLIMITS are used to give PPLUS a range to draw).

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	INTEGER cx, ndim, dim(nferdims)

* local variable declarations:
	INTEGER	CX_DIM_LEN_ZERO, idim, ii, clen(nferdims)

* initialize
	ndim = 0
	DO idim = 1, nferdims
 	   clen(idim) = CX_DIM_LEN_ZERO( idim, cx )
 	ENDDO

* find the axes with a range of values
	DO idim = 1, nferdims
	   IF ( clen(idim) .GT. 1 ) THEN
	      ndim = ndim + 1
	      dim(ndim) = idim
	   ENDIF
 	ENDDO

* find the axes where the context is a single point
	ii = ndim
	DO idim = 1, nferdims
	   IF ( clen(idim) .EQ. 1) THEN
	      ii = ii + 1
	      dim(ii) = idim
	   ENDIF
 	ENDDO

* find the axes where the context is no points

	DO idim = 1, nferdims
	   IF ( clen(idim) .EQ. 0 ) THEN
	      ii = ii + 1
	      dim(ii) = idim
	   ENDIF
 	ENDDO

	RETURN	   
	END
