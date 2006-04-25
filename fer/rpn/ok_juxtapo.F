	LOGICAL FUNCTION OK_JUXTAPO (las_ptr,inf_ptr,atom_type,atom_id)

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
*	Determines whether juxtaposition of two atom types in the infix string
*	is allowed in the syntax of things.

*	J Davison TMAP/JISAO/PMEL/NOAA
*	10.22.87
* V200: 11/23/89 - replaced constants with parameter names and strings !!!
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA rather than PARAMETER stmnt.
* V450: 12/96 - add checks for grid changing functions and child variables
* V500: 12/98 *sh*  added alg_const_var -- for fixed arrays such as "{1,3,5}"
*	 1/99 - added alg_string   
* V554 1/04 *acm* Add category alg_counter_var for counter variables; REPEAT/RANGE=/NAME=
* V600 8/05 *acm* - Add variable type attrib_val for definitions of the form
*                   varname.attname

	include		'ferret.parm'
	include		'rpn.parm'
	include		'xalgebra.cmn'

	CHARACTER	punct*1
	integer		las_ptr,inf_ptr,atom_type(*),atom_id(*)
	integer		atom(2),ndx(2),i,val
	integer		kar_matrix(8,8)
	DATA	 	kar_matrix	/0,0,1,0,0,1,0,0,
     .					1,1,0,1,1,0,1,0,
     .					1,1,0,1,1,0,1,0,
     .					0,0,1,0,0,1,0,0,
     .					1,1,0,1,1,0,1,1,
     .					0,0,1,0,0,1,0,0,
     .					0,0,1,1,1,1,0,0,
     .					1,1,0,1,1,0,1,0/

*******************************************************************************

	atom(1) = las_ptr
	atom(2) = inf_ptr

	do 100 i = 1,2

*	   LEFT_PAREN PLACED ON LEFT END OF INFIX STRING
	   if (atom(i) .eq. added_left_paren) then
	      ndx(i) = 5
	      goto 100
	   end if

*	   ALL OPERATORS EXCEPT NOT
	   if (atom_type(atom(i)) .eq. alg_operator) then
	      if (alg_op(atom_id(atom(i))) .ne. 'NOT' )then
		 ndx(i) = 1
	      else 
	         ndx(i) = 2
	      end if
	      goto 100
	   end if

*	   CONSTANT, VARIABLE, PSEUDO-VARIABLE OR OPERAND
	   if  (atom_type(atom(i)) .eq. alg_constant
     .	   .or.	atom_type(atom(i)) .eq. alg_variable
     .	   .or.	atom_type(atom(i)) .eq. alg_pseudo_var
     .	   .or.	atom_type(atom(i)) .eq. alg_const_var
     .	   .or.	atom_type(atom(i)) .eq. alg_string
     .	   .or.	atom_type(atom(i)) .eq. alg_child_var
     .	   .or. atom_type(atom(i)) .eq. alg_operand
     .	   .or. atom_type(atom(i)) .eq. alg_counter_var
     .	   .or. atom_type(atom(i)) .eq. alg_attrib_val ) then
	      ndx(i) = 3
	      goto 100
	   end if

*	   COMMA,LEFT OR RIGHT PARENTHESIS
	   if (atom_type(atom(i)) .eq. alg_punctuation ) then
	      punct = alg_punct( atom_id(atom(i)) )
	      if ( punct .eq. ',' ) ndx(i) = 4
	      if ( punct .eq. '(' ) ndx(i) = 5
	      if ( punct .eq. ')' ) ndx(i) = 6
	      goto 100
	   end if

*	   WHERE,THEN OR ELSE
	   if (atom_type(atom(i)) .eq. alg_log_struct) then
	      ndx(i) = 7
	      goto 100
	   end if

*	   FUNCTION
	   if (atom_type(atom(i)) .eq. alg_function
     .	  .OR. atom_type(atom(i)) .eq. alg_grid_chg_fcn) then
	      ndx(i) = 8
	      goto 100
	   end if

*	   ALG_FINISHED (INTERPRETED AS A RIGHT PARENTHESIS)
	   if (atom_type(atom(i)) .eq. alg_finished) then
	      ndx(i) = 6
	      goto 100
	   end if

	   stop 'unknown atom type encountered in ok_juxtapo'
100	continue

	val = kar_matrix(ndx(1),ndx(2))	   

	if (val .eq. 0) then
	   ok_juxtapo = .false.
	else
	   ok_juxtapo = .true.
	end if

	return 
	end
