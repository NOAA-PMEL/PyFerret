	SUBROUTINE GET_WORK_MR ( i8_size, mr, status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* allocate and return a managed block of memory without a corresponding
* context that points to it. 
* The routines delete_variable and delete_uncached_vars are responsible for
* freeing the memory
* This routine may be called to get as many separate chunks of work space
* as are required -- unlike GET_MEM_SPC, which returns always the same
* single hunk of memory. (This routine is superior to GET_WORK_SPC in almost
* every way ... should really replace that routine.)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V510 8/99 *sh*
*      2/00 *sh* bug fix - initialize mr_grid
* V533 7/01 *sh* - set data type
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V69+ *acm*  9/14 Ticket 1758. Memory request uses number of blocks not words

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	mr, status
	INTEGER*8 i8_size

* internal variable declarations
	INTEGER idim

* --- end of introductory code ---

* get a free slot in the memory variable table
	CALL FIND_MR_SLOT( mr, status )
	IF ( status .NE. ferr_ok ) RETURN

* allocate memory
	CALL GET_MR_DYNMEM( i8_size, mr, status )
	IF ( status .NE. ferr_ok ) THEN
	   CALL MR2FREELIST(mr)
	   RETURN
	ENDIF

* always set its data type to float
	mr_type( mr ) = ptype_float

* always set its missing value flag to the default
	mr_bad_data( mr ) = bad_val4

* flag it as "in progress"
	mr_protected( mr )	= mr_in_progress

* book keeping about memory usage
	essential_mem = essential_mem + mr_size(mr)
	IF (essential_mem.GT.peak_essential_mem) 
     .		peak_essential_mem = essential_mem

* work arrays do not use any grids
	mr_grid( mr ) = unspecified_int4

* for good house-keeping, set the mr subscripts
	mr_lo_ss(mr,x_dim) = 1
	mr_hi_ss(mr,x_dim) = i8_size
	DO 100 idim = 2, nferdims
	   mr_lo_ss(mr,idim) = unspecified_int4
	   mr_hi_ss(mr,idim) = unspecified_int4
 100	CONTINUE

* hook it into the cacke search list (just to be consistent with other vars)
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_category(cx_stack_ptr) = cat_work_mr
	CALL HOOK_MR_SLOT( cx_stack_ptr, mr )
	cx_stack_ptr = cx_stack_ptr - 1

	RETURN
	END
