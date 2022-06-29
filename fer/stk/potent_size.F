	INTEGER*8 FUNCTION POTENT_SIZE ( cx )

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
* determine the "potential" size of context cx - that is, the total number of
* values that will need to be accessed in order to perform the indicated
* compressing transformations (transf. that reduce axis to a point)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/8/89
*	8/29/89 - special treatment of interpolation
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*       *acm* 5/22 Issue 112: POTENT_SIZE fcn must be integer*8 to represent total 
*                  size of large grids
 
        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	INTEGER	idim, status
	INTEGER*8 CGRID_SIZE8

* make a copy of the context in a scratch buffer
	CALL TRANSFER_CONTEXT( cx, cx_buff )

* expand all the compressing transformations
	DO 100 idim = 1, nferdims
	   IF ( cx_trans(idim,cx) .EQ. trans_interpolate ) THEN
	      cx_trans(idim,cx_buff) = trans_no_transform
	      cx_lo_ss(cx_buff,idim) = 1
	      cx_hi_ss(cx_buff,idim) = 2
	   ELSEIF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	      cx_trans(idim,cx_buff) = trans_no_transform
	      cx_by_ss(idim,cx_buff) = .FALSE.
	      CALL FLESH_OUT_AXIS( idim, cx_buff, status )	! ignore status
	   ENDIF
 100	CONTINUE

* now, what's the size of the result ?

	POTENT_SIZE = CGRID_SIZE8( cx_buff )

	RETURN
	END
