	SUBROUTINE AUX_REGRID_LIMS( dst_cx, src_cx, axis_list, naxis,
     .			aux1, mr1, cx1,
     .			aux2, mr2, cx2,
     .			aux3, mr3, cx3,
     .			status )

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
* Given a context describing data to be regridded to a destination grid
* create the description of the needed source data in the source context
* Use the auxiliary variables to determine the index range of data needed
* For example, if the regrid has requested depth from 0 to 100 and in the 
* auxiliary variable containing depths all of the values LE 100
* are in K=1:6 then we need only request source data over the range K=1:7 

* When called the source context should be a copy of the destination context 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*V685  4/13 *sh*
*V689+ 3/15 *sh* added T-axis F-T orthoganalization

	IMPLICIT NONE
	include	'ferret.parm'
	include	'errmsg.parm'

* calling argument declarations:
	INTEGER	src_cx, axis, dst_cx, naxis, axis_list(*), status,
     .		mr1, cx1, mr2, cx2, mr3, cx3 
        REAL*8  aux1(*), aux2(*), aux3(*)


* this is just a dispatch routine that fans out to routines
* based upon the number of dimensions in the regridding

	IF ( naxis .EQ. 1 ) THEN
	   IF ( axis_list(1) .EQ. z_dim ) THEN
	      CALL AUX_REGRID_LIMS_1D_Z( dst_cx, src_cx, axis_list, naxis,
     .			                 aux1, mr1, cx1, status )
	   ELSEIF ( axis_list(1) .EQ. t_dim ) THEN
	      CALL AUX_REGRID_LIMS_1D_T( dst_cx, src_cx, axis_list, naxis,
     .			                 aux1, mr1, cx1, status )
	   ELSE
	      GOTO 5100
	   ENDIF
	   IF (status .NE. ferr_ok) GOTO 5000

	ELSEIF (naxis .EQ. 2) THEN
	   IF ( axis_list(1).EQ.t_dim .AND. axis_list(2).EQ.f_dim) THEN
	      CALL AUX_REGRID_LIMS_2D_TF( dst_cx, src_cx, axis_list, naxis,
     .			                 aux1, mr1, cx1, status )
	   ELSE
	      CALL ERRMSG(ferr_internal, status, 'aux_regrid_lims_2d', *5000 )
	   ENDIF

	ELSE
	   CALL ERRMSG(ferr_internal, status, 'aux_regrid_lims_Nd', *5000 )
	ENDIF

* success
        status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .     '1-dim auxiliary variable regrid allowed'//
     .	   ' on Z and T axes only', *5000)
	END
