	SUBROUTINE NOT_RPN (text,atom_type,atom_id,atom_start,atom_end,
     .			    nexp,exp_start,exp_end,exp_at1,exp_atn,
     .			    status,err_msg)

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
* Scan an expression given in Reverse Polish order for illegal atom types
* and break up the expression if there are commas
* other error checking is done at run time

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 11/29/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/rpn.parm'
	include 'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:RPN.PARM'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER		atom_type(maxatoms),atom_id(maxatoms),
     .			atom_start(maxatoms),atom_end(maxatoms),
     .			exp_start(max_exp),exp_end(max_exp), nexp,
     .			exp_at1  (max_exp),exp_atn(max_exp),status
	CHARACTER*(*)	text,err_msg

* internal variable declarations:
	INTEGER		atom

* initialize
	atom = 0
	nexp = 1
	exp_start(1) = atom_start(1)
	exp_at1(1)   = 1

 10	atom = atom + 1
	IF ( atom_type(atom) .EQ. alg_finished ) THEN
	   GOTO 500
	ELSEIF ( atom_type(atom) .EQ. alg_log_struct ) THEN
	   GOTO 5100
	ELSEIF ( atom_type(atom) .EQ. alg_punctuation ) THEN
	   IF ( atom_id(atom) .EQ. comma ) THEN
	      GOTO 100
	   ELSE
	      GOTO 5100
	   ENDIF
	ELSEIF ( atom_type(atom) .EQ. alg_constant
     .	   .OR.  atom_type(atom) .EQ. alg_variable
     .	   .OR.  atom_type(atom) .EQ. alg_pseudo_var
     .	   .OR.  atom_type(atom) .EQ. alg_function
     .	   .OR.  atom_type(atom) .EQ. alg_operator ) THEN
	   GOTO 10	
	ENDIF
	STOP 'NOT_RPN'

* comma - end of current and beginning of new expression
 100	exp_end(nexp) = atom_end(atom-1)
	exp_atn(nexp) = atom-1
	nexp = nexp + 1
	IF ( nexp .GT. max_exp ) GOTO 5200
	exp_start(nexp) = atom_start(atom+1)	! ... limited error check
	exp_at1  (nexp) = atom+1
	GOTO 10

* done
 500	exp_end(nexp) = atom_end(atom-1)
	exp_atn(nexp) = atom-1

* success
	status = ferr_ok
	RETURN

* error exits
 5000	status = ferr_syntax
	RETURN
 5100	err_msg = 'Illegal Reverse Polish item: '//
     .		   text(atom_start(atom):atom_end(atom))
	GOTO 5000
 5200	err_msg = ' '
	GOTO 5000

	END
