	SUBROUTINE GRP_CX_DOC( cx_list, ncx, joint_cx, flags )

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
* set up joint context and document flags to summarize the contexts
* of one or more variables that will be plotted (or listed) simultaneously
* For example, PLOT U[Z=20],U[Z=50] will place data set and X,Y,T locations
* in cx_plot and set up key_doc to display the Z location
* This routine is called at the start of each fresh (non-overlay) plot

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V230:  7/22/92 - added overlay argument to ADDL_CX_DOC

#ifdef unix
	include	'ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	LOGICAL	flags(0:10)
	INTEGER	ncx, cx_list(ncx), joint_cx

* internal variable declarations
	INTEGER	i, idim

* internal parameter declarations:
        LOGICAL overlay
        PARAMETER ( overlay = .TRUE. )

* initialize the key documentation flags
	DO 100 i = 0, 10
 100	flags(i) = .FALSE.

* the context of the first variable is the starting point
	CALL TRANSFER_CONTEXT( cx_list(1), joint_cx )

* merge the other contexts with it
	DO 200 i = 2, ncx
 200	CALL ADDL_CX_DOC( joint_cx, cx_list(i), .NOT.overlay, flags )

* regridding is always displayed on the key, not the joint context
!	cx_unstand_grid( cx_plot ) = .FALSE.
	RETURN
	END
