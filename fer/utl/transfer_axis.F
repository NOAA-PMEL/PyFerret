	SUBROUTINE TRANSFER_AXIS ( idim, source, dest )

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
* transfer region and transformation information about one axis from one
* context buffer to another

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/31/89 - extracted from TRANSFER_CONTEXT
* $Id$ 
*       - *kob* need to now copy cx_calender as well
* v580 *acm* 11/04 transfer cx_cal_id, the calendar ID used when  
*                  computing world coordinates

	include 'ferret.parm'
	include 'xcontext.cmn'

	INTEGER	source, dest, idim

	cx_lo_ss ( dest, idim )   = cx_lo_ss ( source, idim )
	cx_hi_ss ( dest, idim )   = cx_hi_ss ( source, idim )
	cx_lo_ww ( idim, dest )   = cx_lo_ww ( idim, source )
	cx_hi_ww ( idim, dest )   = cx_hi_ww ( idim, source )
	cx_trans ( idim, dest )   = cx_trans ( idim, source )
	cx_by_ss ( idim, dest )   = cx_by_ss ( idim, source )
	cx_given ( idim, dest )   = cx_given ( idim, source )
	cx_trans_arg(idim, dest)  = cx_trans_arg(idim, source)
* 8/97 *kob* 	
	IF (idim .EQ. t_dim) THEN
           cx_calendar(dest) = cx_calendar(source)
           cx_cal_id(dest) = cx_cal_id(source)
        ENDIF


	RETURN
	END
