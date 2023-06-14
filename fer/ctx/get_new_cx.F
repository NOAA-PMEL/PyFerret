	SUBROUTINE GET_NEW_CX ( dflt_cx, new_cx, apply_cx, status )

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
* using the default context supplied and the context modifiers given with the
* command produce a new context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/22/87 - code stoled from GET_CMND_DATA
* revision 0.1 - 7/27/88 - added dflt_cx to get_context_mods call
* V200:  5/19/89 - transforms not permitted on command line
* V530:  8/00 *sh* - added data type sensitivity
* V600 *acm* 3/06  fixing bugs nferdims39&1390, use apply_cx to say whether
*                  to apply command context to grave-accent expressions.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'

* calling argument declarations
	INTEGER		dflt_cx, new_cx, status
        LOGICAL         apply_cx

* local variable declarations
	INTEGER		idim, cmnd_mods_cx
	CHARACTER	ALG_TRANS_CODE*3

* de-allocate old context space
	cx_stack_ptr = cx_stack_ptr_base

* start with the default context for the evaluation of this expression
	CALL TRANSFER_CONTEXT ( dflt_cx, new_cx )

* create a clean stack slot to receive the command context modifications
	cmnd_mods_cx = cx_stack_ptr + 1
	CALL INIT_CONTEXT_MODS ( cmnd_mods_cx )

* translate the command context modifiers into the modifications buffer
	CALL GET_CONTEXT_MODS ( dflt_cx,
     .				cmnd_buff,
     .				cmnd_mods_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				unknown_qual_ok,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5000

* transformation given on command line ?
	DO idim = 1, nferdims
 	   IF (cx_trans(idim,cmnd_mods_cx) .NE. trans_no_transform)
     .         GOTO 5010
     	ENDDO

        IF (apply_cx) THEN

* modify the command context with the given modifiers
	   CALL APPLY_CONTEXT_MODS
     .			( cmnd_mods_cx, new_cx, replace_context, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
        ENDIF

* set default data type

	cx_type(new_cx) = unspecified_int4

	RETURN

* error exit
 5000	RETURN
 5010	CALL ERRMSG( ferr_syntax, status,
     .	'transforms not permitted on command: @'//
     .		ALG_TRANS_CODE( cx_trans(idim,cmnd_mods_cx) ), *5000 )
	END
