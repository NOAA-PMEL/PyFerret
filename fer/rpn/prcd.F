	INTEGER FUNCTION PRCD (ndx,ptr,atom_type,atom_id)

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
*	Function to determine precedence values of atoms in the infix
*	string and the stack

*	J Davison TMAP/JISAO/PMEL/NOAA
*	10.22.87
* V200: 11/23/89 - changed constants for parameter names !!!
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA rather than PARAMETER stmnt.
* V450: 12/96 - add check for grid changing functions
* V500: 12/98 *sh* - add check for alg_const_var (like "{1,3,5}")
*	 1/99 *sh* - add check for alg_string

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include		'ferret_cmn/rpn.parm'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE		'FERRET_CMN:RPN.PARM'
#endif

	integer 	ndx,ptr,atom_type(*),atom_id(*)

	integer		i
	integer		lp_prc(2)
	DATA 		lp_prc/23,0/
	integer		op_prc(2,13)
	DATA		op_prc/11,12,11,12,13,14,13,14,16,15,5,6,
     .			3,4,9,10,9,10,9,10,9,10,9,10,7,8/
	integer		punc_prc(2,3)
	DATA		punc_prc/0,-999,23,0,0,-999/
	integer		atom_prc(2,alg_string) ! (2,14->16) for V500 

! V450: 12/96 - added initializations for alg_grid_chg_fcn and alg_child_var
	DATA	(atom_prc(i,alg_operator),i=1,2)/-888,-888/,
     .		(atom_prc(i,alg_constant),i=1,2)/21,22/,
     .		(atom_prc(i,alg_variable),i=1,2)/21,22/,
     .		(atom_prc(i,alg_punctuation),i=1,2)/-888,-888/,
     .		(atom_prc(i,alg_function),i=1,2)/17,18/,
     .		(atom_prc(i,alg_pseudo_var),i=1,2)/21,22/,
     .		(atom_prc(i,alg_log_struct),i=1,2)/1,2/,
     .		(atom_prc(i,alg_operand),i=1,2)/21,22/,
     .		(atom_prc(i,alg_finished),i=1,2)/0,-999/,
     .		(atom_prc(i,alg_grid_chg_fcn),i=1,2)/17,18/,
     .		(atom_prc(i,alg_child_var),i=1,2)/21,22/,
     .		(atom_prc(i,alg_const_var),i=1,2)/21,22/,
     .		(atom_prc(i,alg_string),i=1,2)/21,22/

********************************************************************************

*	IS THIS THE LEFT PARENTHESIS PUT ON STACK AT BEGINNING?
	if (ptr .eq. added_left_paren) then
	   prcd = lp_prc(ndx)
	   return
	end if

*	IF AN OPERATOR, DETERMINE PRECEDENCE FROM OP_PRC
	if (atom_type(ptr) .eq. alg_operator) then
	   prcd = op_prc(ndx,atom_id(ptr))
	   goto 100
	end if

*	IF A PUNCTUATION ATOM, FROM PUNC_PRC
	if (atom_type(ptr) .eq. alg_punctuation) then
	   prcd = punc_prc(ndx,atom_id(ptr))
	   goto 100
	end if

*	OTHERWISE FROM ATOM_PRC
	prcd = atom_prc(ndx,atom_type(ptr))

*	CHECK FOR ERRORS
100	if (prcd .eq. -888) stop 'precedence function is fatally flawed'
	if (prcd .eq. -999) stop
     .		'right paren, comma, or alg_finished on stack'
	return

	end
