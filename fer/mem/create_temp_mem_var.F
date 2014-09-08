	SUBROUTINE CREATE_TEMP_MEM_VAR( cx, mr, status )

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
* allocate and claim sufficient memory space, and assign and fill a slot in the 
* memory variable table for the variable or expression indicated in the context
* -- flag the variable as temporary

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/14/88 - from CREATE_MEM_VAR ver 0.0
* V530   8/00 *sh* - double the memory allocation for string variables
*			(to support 64 bit pointers)
* V533   6/01 *sh* - use CGRID_SIZE_DELTA to allow mem vars with delta values
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V69+ *acm*  9/14 Ticket 1758. Memory request uses number of blocks not words

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER		cx, mr, status

* internal variable declarations:
	INTEGER	CGRID_SIZE_DELTA,
     .		grid_size, start_block, num_blocks, grid_blocks
	REAL R_CGRID_SIZE_DELTA, rgrid_size

* amount of storage space required
* Compute as a real - grid sizes may overflow integer storage.

	rgrid_size = R_CGRID_SIZE_DELTA( cx )  ! as a real

* ... note: later we will want an array of word sizes indexed by type
#ifdef double_p
	IF (cx_type(cx) .EQ. ptype_string) rgrid_size = rgrid_size
#else
	IF (cx_type(cx) .EQ. ptype_string) rgrid_size = rgrid_size*2 ! 8 byte ptr
#endif

* claim the storage space

* COMPUTE # BLOCKS NEEDED
	grid_blocks = ( INT(rgrid_size) + mem_blk_size - 1 ) / mem_blk_size

	CALL GET_MEMORY( INT(rgrid_size), grid_blocks, start_block, num_blocks, status )
	IF ( status .NE. ferr_ok ) RETURN

* catalog it in the memory variable table
	CALL ASSIGN_TEMP_VAR
     .		      ( cx,
     .			start_block,
     .			num_blocks,
     .			mr,
     .			status 	)
	IF ( status .NE. ferr_ok ) RETURN

	RETURN
	END
