	SUBROUTINE RPN ( text,
     .			 at_typ_in,atom_id,atom_start,atom_end,
     .			 natom_infx, natom_pstfx, pstfx_order,
     .			 status,err_msg)

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
*	Routine to take an infix expression in the character variable text and 
*	convert to a postfix format.

*	J Davison TMAP/JISAO/PMEL/NOAA
*	10.20.87
* V200: 11/20/89 - *sh* changed include directories
*		- fixed bug in expr_start,expr_end to give pre-RPN string ends
*		- added arguments exp_at1, exp_atn 1st and last atom per expr'n
*		- eliminate nesting of WHERE-THEN-ELSE but allow W-T w/out ELSE
*	 3/20/90 - beef up error message resulting from monadic "-"
*	  5/3/90 - bug fix: exp_ndx max check
* V450	   11/96 - major reorg - this routine now works on only a single
*		   expression at a time and returns the permutation array
*		 - Also, support for alg_grid_chg_fcn
* Linux Port - kob- 4/97 - logic change in if statement to prevent an overflow
*			   error when looking at atom_id array.
* V532 5/01 *sh* - improve error message about unknon function in FCN(a)
* V533 7/01 *sh* - insert alg_if_marker and alg_then_marker instead of no_op
*		   to enable IF cond THEN string1 ELSE string2 
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid. 
* V720: 4/17 *acm* add direction-changing functions as a subset of grid-changing ones
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'rpn.parm'
	include 'xalgebra.cmn'

	integer		at_typ_in(maxatoms),atom_id(maxatoms),
     .			atom_start(maxatoms),atom_end(maxatoms),
     .			natom_infx, natom_pstfx,
     .			pstfx_order(maxatoms),
     .			status

	character*(*)	text,err_msg

	integer		PRCD, GCF_NUM_ARGS, atom_type(maxatoms),
     .			inf_ptr,pos_ndx,stk_ndx,stack(maxatoms),
     .			fcn_ndx,i,arg_cnt(max_fcn),
     .			atom_cnt,inf_prc,tmp_ptr,las_ptr,
     .			ndx, narg,
     .			wte_ndx,wte_cnt(max_wte+1)
	
	logical		OK_JUXTAPO

********************************************************************************

*	INITIALIZE ALL POINTERS AND INDICES
	inf_ptr		= added_left_paren
	pos_ndx 	= 0
	stk_ndx		= 1
	stack(stk_ndx)	= added_left_paren
	
	fcn_ndx		= 0
	do i = 1,max_fcn
	   arg_cnt(i) 	= 1
	enddo

	wte_ndx 	= 1
	do i = 1,max_wte+1
	   wte_cnt(i)   = 0
	enddo

*	TRANSFER THE ATOM TYPE LIST TO A WORKING ARRAY WITH TERMINATOR (11/96)
	DO ndx = 1, natom_infx
	   atom_type(ndx) = at_typ_in(ndx)
 	enddo

	atom_cnt = natom_infx + 1
	atom_type(atom_cnt) = alg_finished	! terminator

*	EXAMINE EACH ATOM IN INFIX STRING, PUT ON STACK OR IN POSTFIX STRING
40	do ndx = 1, atom_cnt
	   las_ptr = inf_ptr
	   inf_ptr = ndx

	   if (.not. OK_JUXTAPO (las_ptr,inf_ptr,atom_type,atom_id)) goto 5200
	   if (atom_type(inf_ptr) .eq. alg_grid_chg_fcn
     .	  .OR. atom_type(inf_ptr) .eq. alg_dir_chg_fcn
     .	  .OR. atom_type(inf_ptr) .eq. alg_function )
     .						fcn_ndx = fcn_ndx + 1
	
	   if (atom_type(inf_ptr) .eq. alg_log_struct .OR.
     .         atom_type(inf_ptr) .EQ. alg_log_vstruct ) then

*	      KEEP TRACK OF WHERE-THEN-ELSE CLAUSE VALIDITY
* ... when "WHERE" is encountered increase the nesting level, wte_ndx
* *kob* - linux port 4/97 change logic from ( .NOT.(A .OR. B) to 
*  	                  ( .NOT.A .AND. .NOT.B )
	      if (atom_id(inf_ptr) .eq. where) then
		 if (.not. (las_ptr  .eq. added_left_paren)) THEN
	      	    if (.not. (atom_type(las_ptr) .eq. alg_punctuation
     .		    .and.
     .		    atom_id(las_ptr) .eq. left_paren)) goto 5100
		    endif
		 wte_ndx = wte_ndx + 1
	         IF (wte_ndx .GT. 2 ) GOTO 5150
	      end if

* ... bump the WHERE-THEN-ELSE pointer, wte_cnt, is it the right logic item ?
	      wte_cnt(wte_ndx) = wte_cnt(wte_ndx) + 1
	      if (wte_cnt(wte_ndx) .ne. atom_id(inf_ptr)) 	     goto 5100

* ... decrement the W-T-E nesting level when ELSE is encountered
!	      if (wte_cnt(wte_ndx) .eq. else) then
!	         wte_cnt(wte_ndx) = 0
!	         wte_ndx          = wte_ndx - 1
!	      	 if (wte_ndx .lt. 1) 				     goto 5100
!	      end if
	   end if

	   inf_prc = PRCD (inf_flg,inf_ptr,atom_type,atom_id)
50	   if (inf_prc .gt. PRCD (stk_flg,stack(stk_ndx),atom_type,atom_id))
     .								          then
*	      PUT THIS ATOM ONTO STACK AND CONTINUE 
	      stk_ndx 	     = stk_ndx + 1
	      stack(stk_ndx) = inf_ptr
	   else
*	      POP OFF THE TOP OF THE STACK AND LOOK AT IT
	      tmp_ptr = stack(stk_ndx)
	      stk_ndx = stk_ndx - 1
	      if (stk_ndx .eq. 0) then
	         if (.not.( (inf_ptr .eq. atom_cnt) .or. 
     .		            (atom_type(inf_ptr) .eq. alg_punctuation .and. 
     .		             atom_id  (inf_ptr) .eq. comma) ))       goto 5300
	      end if

*	      PLACE ATOM OF HIGHER PRECEDENCE IN POSTFIX STRING
	      if (inf_prc .lt. PRCD (stk_flg,tmp_ptr,atom_type,atom_id))  then
		 pos_ndx	 = pos_ndx + 1 
		 pstfx_order(pos_ndx) = tmp_ptr

*		 ENSURE FUNCTION HAS RIGHT # OF ARGUMENTS
	         if (atom_type(tmp_ptr) .eq. alg_function) then
*                   ==> standard internal Ferret function
	            narg = alg_num_fcn_args(atom_id(tmp_ptr))
		    if (arg_cnt(fcn_ndx) .ne. narg)  goto 5400
		    arg_cnt(fcn_ndx) = 1
		    fcn_ndx 	     = fcn_ndx - 1
	         ELSEif (atom_type(tmp_ptr) .eq. alg_grid_chg_fcn .OR.
     .		         atom_type(tmp_ptr) .eq. alg_dir_chg_fcn ) then
*                   ==> grid changing Ferret function
*		        variable arguments?
	            narg = GCF_NUM_ARGS( atom_id(tmp_ptr) )
	            if ( narg .GE. 0 ) THEN
		       if (arg_cnt(fcn_ndx) .ne. narg)  goto 5400
	            else
		       if (arg_cnt(fcn_ndx) .LT. ABS(narg))  goto 5400
	            ENDIF
		    arg_cnt(fcn_ndx) = 1
		    fcn_ndx 	     = fcn_ndx - 1
	         end if

! 11/96 - obsolete lines dealing with multiple expressions
!		 if (exp_end(exp_ndx) .ne. 0) then
!*		    LAST EXPRESSION IS COMPLETE AND NEW ONE IS BEING BUILT
!		    exp_ndx	       = exp_ndx + 1
!	            IF ( exp_ndx .GE. max_exp ) GOTO 5600	! fix *sh* 5/90
!!		    exp_start(exp_ndx) = atom_start(pstfx_order(pos_ndx)) ! old
!* ... exp_start for expr'n n is determined at the end of expr'n n-1
!		    exp_at1  (exp_ndx) = pos_ndx	! 1st atom of RPN
!		 end if
		 goto 50
	      else
*	         PARENTHESIS MATCH OR LEFT PARENTHESIS AND A COMMA

	         if (atom_type(inf_ptr) .eq. alg_punctuation .and.
     .               atom_id  (inf_ptr) .eq. comma)			  then

*	            RETURN LEFT PARENTHESIS TO STACK
	            stk_ndx	   = stk_ndx + 1
	            stack(stk_ndx) = tmp_ptr

*	            COMMA INDICATES NEW FUNCTION ARGUMENT OR END OF EXPRESSION
	            if (fcn_ndx .ne. 0) then
		       arg_cnt(fcn_ndx) = arg_cnt(fcn_ndx) + 1
	            else
	               GOTO 5600
! 11/96 - obsolete lines dealing with multiple expressions
!!		       exp_end(exp_ndx) = atom_end(pstfx_order(pos_ndx)) ! old
!		       exp_end(exp_ndx) = atom_end(las_ptr)
!		       exp_atn(exp_ndx) = pos_ndx	! last atom of RPN
!	               exp_start(exp_ndx+1) = atom_start(inf_ptr+1)
	            end if
	         end if 
	      end if
	   end if

	enddo

! 11/96 - obsolete lines dealing with multiple expressions
!!	exp_end(exp_ndx) = atom_end(pstfx_order(pos_ndx))	! old
!	exp_end(exp_ndx) = atom_end(atom_cnt-1)
!	exp_atn(exp_ndx) = pos_ndx			! last atom of RPN

	if (wte_ndx .gt. 1) then
* ... "WHERE" is a no-op at RPN execution time
	   DO i = 1, atom_cnt
	     IF ( ((atom_type(i) .EQ. alg_log_struct) .OR. 
     .             (atom_type(i) .EQ. alg_log_vstruct) )
     .	      .AND. atom_id(i)   .EQ. where ) at_typ_in(i) = alg_then_marker
     	   ENDDO

	   IF (wte_cnt(2) .eq. where) THEN
	      goto 5100
	   ELSEIF ( wte_cnt(2) .EQ. pelse ) THEN

* ... WHERE-THEN-ELSE --> THEN becomes a no-op
	      DO i = 1, atom_cnt
	         IF ( ((atom_type(i) .EQ. alg_log_struct) .OR. 
     .                 (atom_type(i) .EQ. alg_log_vstruct) )
     .	          .AND. atom_id(i)   .EQ. pthen ) at_typ_in(i) = alg_if_marker
     	      ENDDO

	   ENDIF
	end if

	if (stk_ndx .eq. 0) then

*	DONE - RETURN THE pstfx_order ARRAY
	   natom_pstfx = pos_ndx
	   status = ferr_ok
	   return
	else
	   goto 5500
	end if

** error exits
 5000	status = ferr_syntax
	return

*	WHERE-THEN-ELSE CLAUSE ORDER IS INCORRECT
 5100	err_msg = 'There is an error in if-then-else clause order'
	GOTO 5000

*	WHERE-THEN-ELSE CLAUSES ARE NESTED
 5150	err_msg = 'Cannot use multiple IFs in an expression'
	GOTO 5000


*	DISALLOWED JUXTAPOSITION OF ATOM TYPES AT LAS_PTR & INF_PTR
 5200	if (las_ptr .eq. added_left_paren) then
     	   err_msg = text(atom_start(inf_ptr):atom_end(inf_ptr))//
     .	   ' may not start an expression'  
	else if (inf_ptr .eq. atom_cnt) then
     	   err_msg = text(atom_start(las_ptr):atom_end(las_ptr))//
     .	   ' may not end an expression'  
	else if (atom_type(inf_ptr).eq.alg_punctuation
     .     .and. atom_id  (inf_ptr).eq.left_paren       ) then
     	   err_msg = 'unknown function "'//
     .			text(atom_start(las_ptr):atom_end(las_ptr))//'"'
	else
	   err_msg = text(atom_start(las_ptr):atom_end(las_ptr))//' before '
     .	   //text(atom_start(inf_ptr):atom_end(inf_ptr))//' is illegal'  
	   IF (text(atom_start(inf_ptr):atom_end(inf_ptr)).EQ.'-' ) THEN
	      i = (atom_end(las_ptr)-atom_start(las_ptr)+1)
     .		+ (atom_end(inf_ptr)-atom_start(inf_ptr)+1) + 19
	      IF ( inf_ptr+1 .EQ. atom_cnt ) THEN
	         err_msg = err_msg(:i)//pCR//
     .		'and "-" may not end expression'
	      ELSEIF ( atom_type(inf_ptr+1) .EQ. alg_constant ) THEN
	         err_msg = err_msg(:i)//pCR//
     .		'negative constants need to be enclosed in parentheses'
	      ELSE
	         err_msg = err_msg(:i)//pCR//
     .		'Monadic minus sign isnt allowed.  Use "(-1)*"'
	      ENDIF
	   ENDIF
	end if
	GOTO 5000

*	MISMATCHED PARENTHESIS NEAR ATOM INF_PTR
 5300	err_msg = 'There is a mismatch of parentheses near '//
     .	text(atom_start(las_ptr):atom_end(las_ptr))
	GOTO 5000

*       INCORRECT NUMBER OF ARGUMENTS IN FUNCTION TMP_PTR
 5400	err_msg = 'The '//text(atom_start(tmp_ptr):atom_end(tmp_ptr))//
     .  ' function has an incorrect number of arguments'
	GOTO 5000

*	MISMATCHED PARENTHESIS NEAR ATOM STACK(STK_NDX)
 5500	if (stack(stk_ndx) .eq. added_left_paren) then
	   err_msg = 'There is a mismatch of parentheses near '//
     .     text(atom_start(stack(stk_ndx+1)):atom_end(stack(stk_ndx+1)))
	else
	   err_msg = 'There is a mismatch of parentheses near '//
     .     text(atom_start(stack(stk_ndx)):atom_end(stack(stk_ndx)))
	end if
	GOTO 5000
 5600	err_msg = 'Definition contains multiple expressions'
	GOTO 5000

! 5600	write ( err_msg, 5610 ) max_exp
! 5610	FORMAT ('Number of expressions in line must be less than',I4)
!	GOTO 5000
	end
