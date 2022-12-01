	SUBROUTINE OV_GRP_CX_DOC( ov_cx, ncx, flags )

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
* merge additional variable contexts with those already merged into the
* joint context for a plot to produce a list of documentation flags suitable
* to make a key for this new context.  See GRP_CX_DOC

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V230:  7/22/92 - added overlay argument to ADDL_CX_DOC
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V685+ *acm* 12/13 For 6D Ferret, the array coming in as flags is longer for extra dimensions
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xplot_setup.cmn'

* calling argument declarations
	INTEGER ncx, ov_cx(ncx)
	LOGICAL	flags(0:12)

* internal variable declarations
	INTEGER	i

* internal parameter declarations:
        LOGICAL overlay
        PARAMETER ( overlay = .TRUE. )
           
* make a fresh copy of cx_plot and key_doc to keep current with the plot labels
	CALL TRANSFER_CONTEXT( cx_plot, cx_buff )
	DO i = 0, 10
 	   flags(i) = key_doc(i)
 	ENDDO

* merge the additional contexts
	DO i = 1, ncx
 	   CALL ADDL_CX_DOC( cx_buff, ov_cx(i), overlay, flags )
 	ENDDO

	RETURN
	END
