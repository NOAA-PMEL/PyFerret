	SUBROUTINE CD_WRITE_DEFER_COORD( cdfid, status )


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
* Write (deferred) netCDF axis coordinates based upon infermation stored
* in COMMON
* (required for performance since switching between DEFINE and DATA modes
* can be so costly)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530 *sh* - 12/00
*	    -  1/01 bug fix -- upper box bound of last coord wrong
* V570 *acm*   5/04 For irregular axes, write bounds coordinate variable, 
*                   dimensioned 2xN
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V65  *acm* 1/10  Make sure parameters values are consistent with 
*                              whats in netcdf.inc from netcdf4.
* V65  *acm* 2/10- all warnings and notes to std error not std out
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* argument declarations:
	INTEGER cdfid, status

* include files
        include 'netcdf.inc'
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
        include 'tmap_errors.parm'
	include 'xio.cmn_text'
        include 'xdefer_coords.cmn'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* local variable declarations:
	INTEGER	 iax, i, i1000, hilim, cdfstat, box_loc,
     .           start(2), count(2)
	REAL*8   buff(1000)
	REAL*8   TM_WORLD


* put CDF into data mode so coordinates can be written
        CALL CD_SET_MODE(cdfid, pcd_mode_data, status)
        IF ( status .NE. merr_ok ) RETURN

* loop over all of the deferred axes
	DO iax = 1, dfr_num

*   write the coordinates -- buffer 1000 at a time
	  IF (dfr_axis_type(iax) .EQ. pcd_defer_points) THEN
	    box_loc = box_middle
	  ELSEIF (dfr_axis_type(iax) .EQ. pcd_defer_edges) THEN
	    box_loc = box_lo_lim
	  ELSEIF (dfr_axis_type(iax) .NE. pcd_defer_bounds) THEN
	    CALL TM_NOTE( '***IntrnlERR - CD_DEFER_COORD_WRITE',
     .				lunit_errors)
	  ENDIF


* Write coordinates or edges as N by 1 or N+1 by 1

          IF (dfr_axis_type(iax) .EQ. pcd_defer_points .OR.
     .        dfr_axis_type(iax) .EQ. pcd_defer_edges) THEN
             DO i1000 = dfr_lo(iax), dfr_hi(iax), 1000
               hilim = MIN((i1000-1)+1000, dfr_hi(iax))
               DO i = i1000, hilim
                  buff(i-i1000+1) = TM_WORLD(i, dfr_grid(iax), 
     .			              dfr_idim(iax), box_loc)
	       ENDDO
               cdfstat = NF_PUT_VARA_DOUBLE (cdfid, dfr_cdaxis(iax), 
     .		       i1000-dfr_lo(iax)+1, hilim-i1000+1, buff)
               IF (cdfstat .NE. NF_NOERR) GOTO 5010
	    ENDDO

* ... for edge defn tag on the upper bound of the top grid box
	     IF (dfr_axis_type(iax) .EQ. pcd_defer_edges) THEN
                buff(1) = TM_WORLD(dfr_hi(iax), dfr_grid(iax), 
     .			       dfr_idim(iax), box_hi_lim)
               cdfstat = NF_PUT_VAR1_DOUBLE (cdfid, dfr_cdaxis(iax),
     .			dfr_hi(iax)-dfr_lo(iax)+2, buff)
               IF (cdfstat .NE. NF_NOERR) GOTO 5010
	     ENDIF

* Write bounds as N by 2, 1000 sets buffered at a time.

          ELSEIF (dfr_axis_type(iax) .EQ. pcd_defer_bounds) THEN
             DO i1000 = dfr_lo(iax), dfr_hi(iax), 500
                hilim = MIN((i1000-1)+500, dfr_hi(iax))
                buff(1) = TM_WORLD(i1000, dfr_grid(iax), 
     .			              dfr_idim(iax), box_lo_lim)
                DO  i = i1000, hilim
                   buff(2*(i-i1000+1)-1) = TM_WORLD(i, dfr_grid(iax), 
     .			              dfr_idim(iax), box_lo_lim)
                   buff(2*(i-i1000+1)) = TM_WORLD(i, dfr_grid(iax), 
     .			              dfr_idim(iax), box_hi_lim)
     	        ENDDO


                start(1) = 1
                count(1) = 2
                start(2) = i1000-dfr_lo(iax)+1
                count(2) =  hilim-i1000+1

                cdfstat = NF_PUT_VARA_DOUBLE (cdfid, dfr_cdaxis(iax), start,
     .		          count, buff)
               IF (cdfstat .NE. NF_NOERR) GOTO 5010
	     ENDDO
           ENDIF

	ENDDO

	dfr_num = 0   ! just house-keeping to be on the safe side

	RETURN
 5010   status = cdfstat
        RETURN
	END
