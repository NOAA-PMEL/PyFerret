	SUBROUTINE OBJECTIVE( memory, xpts, ypts, zpts,
     .			      mr_list, cx_list, nvars,
     .			      opt1_string, opt2_string,
     .			      fname, fmt, status )

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
* perform an objective analysis of 3 arrays, xpts, ypts, zpts, onto a 2D grid
* using the same code as PLOT+ does for scattered arrays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V301: 11/93	USER/COMMAND="objective analysis"
*		    /OPT1=xlo:xhi:xdel,ylo:yhi:ydel,[cay],[rng]
*		    /FILE=filename
*       1/94	added /FORMAT=(FORTRAN format)
* V312: 5/94 - array "memory" as a calling argument
* v314:  8/22/94 Ibm Port *kob* - need to use risc buff for inherited 
*				  length string concats
* V320: 2/95 - allow nrng to be zero

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
#ifdef NO_PASSED_CONCAT	
	include 'xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	nvars, mr_list(nvars), cx_list(nvars), lun, status
	REAL    xpts(*), ypts(*), zpts(*),
     .		memory( mem_blk_size, max_mem_blks )
	CHARACTER opt1_string*(*), opt2_string*(*), fname*(*), fmt*(*)

* internal variable declarations:
	LOGICAL	opt1, opt2
	INTEGER	TM_LENSTR1, CGRID_SIZE, 
     .		loc, i, slen, flen, npts, cx, ndim, dim(4), ip(0:8),
     .		np, nx, ny, nblk, mblk1, nblks, nrng, fmlen
	REAL		p(8), cay
	CHARACTER	c*1, field*24

* Must be exactly 3 arguments given, all of equal size, not necessarily lines
	IF ( nvars .NE. 3 ) GOTO 5200
	npts = unspecified_int4
	DO 100 i = 1, 3
	   cx = cx_list(i)
	   CALL GET_CX_DIMS( cx, ndim, dim ) 
!	   IF ( ndim .NE. 1 ) GOTO 5200
	   IF ( npts .EQ. unspecified_int4 )
     .			npts = CGRID_SIZE( cx )	   
	   IF ( npts .NE. CGRID_SIZE( cx ) ) GOTO 5200
 100	CONTINUE
	IF ( npts .GT. 50000 ) GOTO 5300      !  hard coded in error message!!

* DECODE THE OPT1 COMMAND QUALIFIER:
* /OPT1 = xlo:xhi:xdel,ylo:yhi:ydel,cay,nrng
* ... locate parameter separators in the option string
	slen = TM_LENSTR1( opt1_string )
	np = 0
	ip(0) = 0
	DO 10 i = 1, slen
	   c = opt1_string(i:i)
	   IF ( c .EQ. ',' .OR. c .EQ. ':' ) THEN
	      np = np + 1
	      IF ( np .GE. 8 ) GOTO 5100
	      ip(np) = i
	   ENDIF
 10	CONTINUE
	np = np + 1
	ip(np) = slen+1
* ... decode the parameters from the option string
	DO 20 i = 1, 6
 20	p(i) = bad_val4
	p(7) = 5.		! cay default
	p(8) = 5.		! nrng default
	DO 30 i = 1, np
	   IF ( ip(i)-1 .GE. ip(i-1)+1 ) THEN
	      field = opt1_string(ip(i-1)+1:ip(i)-1)
	      READ (field,'(F24.0)',ERR=5100) p(i)
	   ENDIF
 30	CONTINUE
	DO 40 i = 1, 8
	   IF ( p(i) .EQ. bad_val4 ) GOTO 5100
 40	CONTINUE
	cay = p(7)
	nrng = p(8)
	IF ( nrng .LT. 0 ) GOTO 5100  ! "LE" -> "LT" 2/95

* insert the default filename based on format type
	IF ( fname .EQ. ' ' ) THEN
	   IF ( fmt .EQ. ' ' ) THEN
	      fname = 'fer_objective.unf'
	   ELSE
	      fname = 'fer_objective.dat'
	   ENDIF
	ENDIF
	flen = TM_LENSTR1( fname )
	fmlen = TM_LENSTR1( fmt )

* allocate memory for the job to be done
* need memory for copies of the 3 input arrays
* and for the output 2D array
	nx = (p(2) - p(1))/ABS(p(3)) + 1.9999
	ny = (p(5) - p(4))/ABS(p(6)) + 1.9999
	IF ( nx.LE.0 .OR. ny.LE.0 ) GOTO 5100	
	nblk = (3*npts-1)/mem_blk_size + 1     ! # of blocks for, x,y,z lists
	CALL GET_MEMORY( nblk*mem_blk_size+nx*ny, mblk1, nblks, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* call the FERRET work routine
	CALL OBJECTIVE_SUB(  memory, mr_list, cx_list,  nx, ny, npts,
     .			     cay, nrng, fname(1:flen), fmt(1:fmlen),
     .			     p(1), ABS(p(3)), p(4), ABS(p(6)),
     .			     xpts, ypts, zpts,
     .			     memory(1,mblk1), memory(1,mblk1+nblk),
     .			     status )
     
* cleanup
	CALL FREE_MEMORY( mblk1, nblks )
	RETURN

* error exits
 5000	RETURN
#ifdef NO_PASSED_CONCAT	
	risc_buff = opt1_string(1:slen)
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		    '/OPT1="'//risc_buff//'"'//
     .		    ' is not xlo:xhi:xdel,ylo:yhi:ydel,[cay],[rng]',
     .		    *5000 )
#else
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		    '/OPT1="'//opt1_string(1:slen)//'"'//
     .		    ' is not xlo:xhi:xdel,ylo:yhi:ydel,[cay],[rng]',
     .		    *5000 )
#endif
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		    'must be 3 arrays with equal # of points', *5000 )
 5300	CALL ERRMSG( ferr_prog_limit, status,
     .		    'too many input points - limit 50000', *5000 )

	END

