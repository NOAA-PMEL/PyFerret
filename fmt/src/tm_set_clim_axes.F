	SUBROUTINE TM_SET_CLIM_AXES 

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
*  Allocate dynamic line storage and put coordinate data into lines for climatogical axes


        include 'tmap_dims.parm'
#include "gt_lib.parm"
        include 'xtm_grid.cmn_text'
        external xgt_grid_data


	INTEGER i, npts, status

	npts = 12

* MONTH_IRREG

	CALL GET_LINE_DYNMEM (npts, clmirr, status) 

	DO i = 1, npts
	   CALL PUT_LINE_COORD ( linemem(clmirr)%ptr, i, clim_line_coords(i,clmirr) )
	   CALL PUT_LINE_COORD ( lineedg(clmirr)%ptr, i, clim_line_edges(i,clmirr) )
	ENDDO
	CALL PUT_LINE_COORD ( lineedg(clmirr)%ptr, npts+1, clim_line_edges(i,clmirr) )

    
* MONTH_GREGORIAN
     
	CALL GET_LINE_DYNMEM (npts, clmgrg, status) 
	DO i = 1, npts
	   CALL PUT_LINE_COORD ( linemem(clmgrg)%ptr, i, clim_line_coords(i,clmgrg) )
	   CALL PUT_LINE_COORD ( lineedg(clmgrg)%ptr, i, clim_line_edges(i,clmgrg) )
	ENDDO
	CALL PUT_LINE_COORD ( lineedg(clmgrg)%ptr, npts+1, clim_line_edges(i,clmgrg) )


* MONTH_NOLEAP
          
	CALL GET_LINE_DYNMEM (npts, clmnol, status) 
	DO i = 1, npts
	   CALL PUT_LINE_COORD ( linemem(clmnol)%ptr, i, clim_line_coords(i,clmnol) )
	   CALL PUT_LINE_COORD ( lineedg(clmnol)%ptr, i, clim_line_edges(i,clmnol) )
	ENDDO
	CALL PUT_LINE_COORD ( lineedg(clmnol)%ptr, npts+1, clim_line_edges(i,clmnol) )


* MONTH_ALL_LEAP
    
	CALL GET_LINE_DYNMEM (npts, clmall, status) 
	DO i = 1, npts
	   CALL PUT_LINE_COORD ( linemem(clmall)%ptr, i, clim_line_coords(i,clmall) )
	   CALL PUT_LINE_COORD ( lineedg(clmall)%ptr, i, clim_line_edges(i,clmall) )
	ENDDO
	CALL PUT_LINE_COORD ( lineedg(clmall)%ptr, npts+1, clim_line_edges(i,clmall) )


* MONTH_JULIAN
     
	CALL GET_LINE_DYNMEM (npts, clmjul, status) 
	DO i = 1, npts
	   CALL PUT_LINE_COORD ( linemem(clmjul)%ptr, i, clim_line_coords(i,clmjul) )
	   CALL PUT_LINE_COORD ( lineedg(clmjul)%ptr, i, clim_line_edges(i,clmjul) )
	ENDDO
	CALL PUT_LINE_COORD ( lineedg(clmjul)%ptr, npts+1, clim_line_edges(i,clmjul) )

	RETURN
	END
