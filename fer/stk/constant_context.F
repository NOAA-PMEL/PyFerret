	SUBROUTINE CONSTANT_CONTEXT
     .		      ( const_text, 
     .			default_cx,
     .			const_cx,
     .			status )

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
* return in const_cx context information describing the constant

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/28/87
* revision 0.1 - 5/5/88  - FINISH_CONTEXT_MODS --> FLESH_OUT_CONTEXT
* V200:  6/20/89 - re-worked user defined variable logic 
* V300: ifdef to work around "*" bug in READ (buff,*) in DEC f77 3.2.2
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V751  *acm* 6/19 read directly from const_text instead of putting that
*                  text into a smaller buffer. See issue 1929 allowing
*                  e.g for `00000000000000000012345678901234567890`
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'

* calling argument declarations:
	INTEGER		default_cx, const_cx, status
	CHARACTER*(*)	const_text

* internal variable declarations:
	INTEGER		idim
	REAL		const_buff

* equivalences
	REAL		cx_value( max_cx )
	EQUIVALENCE   ( cx_variable, cx_value )

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, const_cx )
	cx_category( const_cx ) = cat_constant
	cx_bad_data( const_cx ) = bad_val4
	cx_data_set( const_cx ) = pdset_irrelevant
	cx_dset_gvn( const_cx ) = .TRUE.
	cx_type    ( const_cx ) = ptype_float

* get the value of the constant
* bug is that if a "*" follows the numeric text the error branch is taken
	READ ( const_text, *, ERR=5100 ) const_buff
	cx_value( const_cx ) = const_buff

* flesh out context by removing all axes limits
	DO idim = 1, nferdims
	   CALL DEL_CX_DIM( idim, const_cx )	   
	ENDDO

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, const_text, *5000 )
	END
