	SUBROUTINE GET_WORK_MR ( size, mr, status )

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

* calling argument declarations:
	INTEGER	size, mr, status

* internal variable declarations
	INTEGER start_block, num_blocks, idim

* --- end of introductory code ---
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

	CALL GET_MEMORY( size, start_block, num_blocks, status )

* get a free slot in the memory variable table
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_category(cx_stack_ptr) = cat_work_mr
	CALL FIND_MR_SLOT( cx_stack_ptr, mr, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_stack_ptr = cx_stack_ptr - 1

* point mr slot to data in bulk memory
	mr_blk1     ( mr )	= start_block
	mr_nblks    ( mr )	= num_blocks

* always set its missing value flag to the default
	mr_bad_data( mr ) = bad_val4

* flag it as "in progress"
	mr_protected( mr )	= mr_in_progress

* work arrays don't use any grids
	mr_grid( mr ) = unspecified_int4

* for good house-keeping, set the mr subscripts
	mr_lo_ss(mr,x_dim) = 1
	mr_hi_ss(mr,x_dim) = size
	DO 100 idim = 2, 4
	   mr_lo_ss(mr,idim) = unspecified_int4
	   mr_hi_ss(mr,idim) = unspecified_int4
 100	CONTINUE

	RETURN
	END
