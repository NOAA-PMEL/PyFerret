	SUBROUTINE OBJECTIVE_SUB(  memory,
     .			     mr_list, cx_list,  nx, ny, npts,
     .			     cay, nrng, fname, fmt,
     .			     xlo,  xdel, ylo, ydel,
     .			     xpts, ypts, zpts,
     .			     one_d, result, status )

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
* "work routine" to perform an objective analysis of 3 arrays onto a 2D grid
* using the same code as PLOT+ does for scattered arrays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V301: 11/93
*       1/94	added /FORMAT=(FORTRAN format)
* V314: 8/19/94 *kob* IBM port - AIX fortran does not allow ACCESS=APPEND in open
*				 statments.  Added ifdef check for NO_ACCESS_APPEND.
* V312: 5/94 - array "memory" as a calling argument
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* Linux Port 5/97 *kob* - had to add ifdef F90_OPEN_FILE_APPEND because f90
*			  uses "POSITION" instead of "ACCESS" to specify 
*			  opening a file for append.
#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include	'xmem_subsc.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'          ! 12/94 SPLIT_LIST BUFFER
#endif

* calling argument declarations:
	INTEGER	mr_list(3), cx_list(3), nx, ny, npts, nrng, status, ios
	REAL	memory( mem_blk_size, max_mem_blks )
	REAL	xlo,  xdel, ylo, ydel, cay
	REAL    xpts( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		ypts( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		zpts( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		one_d(npts,3), result(nx,ny)
	CHARACTER fname*(*), fmt*(*)

* Note: post-debug can replace x/y/zpts decl with (*)

* internal variable declarations:
	INTEGER	idim, ndim, npts_out, ivar, i, j, flen, mv_temp
	REAL	bad

* initialize
	risc_buff = ' '	! for GUI-redirected output

* extract the 3 variables into a 3-variables  x  npts array, one_d
* ... X locations
	CALL CREATE_TEMP_MEM_VAR( cx_list(1), mv_temp, status )
	IF ( status .NE. ferr_ok ) RETURN
	CALL COPY_GRID( memory(1, mr_blk1(mr_list(1))), mr_list(1),
     .			one_d(1,1), mv_temp )
	CALL DELETE_VARIABLE( mv_temp )
* ... Y locations
	CALL CREATE_TEMP_MEM_VAR( cx_list(2), mv_temp, status )
	IF ( status .NE. ferr_ok ) RETURN
	CALL COPY_GRID( memory(1, mr_blk1(mr_list(2))), mr_list(2),
     .			one_d(1,2), mv_temp )
	CALL DELETE_VARIABLE( mv_temp )
* ... (scattered) values
	CALL CREATE_TEMP_MEM_VAR( cx_list(3), mv_temp, status )
	IF ( status .NE. ferr_ok ) RETURN
	CALL COPY_GRID( memory(1, mr_blk1(mr_list(3))), mr_list(3),
     .			one_d(1,3), mv_temp )
	CALL DELETE_VARIABLE( mv_temp )

* the zgrid routine requires all missing data to be 1.E35
	DO 210 ivar = 1, 3
	   bad = mr_bad_data( mr_list(ivar) )
	   DO 200 i = 1, npts
	      IF ( one_d(i,ivar) .EQ. bad ) one_d(i,ivar)=1.E35
 200	   CONTINUE
 210	CONTINUE

* output grid should be pre-filled with zeros for zgrid
	DO 300 j = 1, ny
	DO 300 i = 1, nx
 300	result(i,j) = 0.0

* do the work
	CALL ZGRID( result, nx, ny, nx, ny, xlo, ylo, xdel, ydel,
     .		    one_d(1,1), one_d(1,2), one_d(1,3), npts, cay, nrng )

* Open the output file
* 8/94 *kob* IBM port --- AIX fortron doesn't allow APPEND
	IF ( fmt .EQ. ' ' ) THEN
	   OPEN( UNIT        = default_list_lun,
     .           FILE        = fname,
     .           FORM        = 'UNFORMATTED',
#ifdef NO_ACCESS_APPEND
#else
#ifdef F90_OPEN_FILE_APPEND
     .		 POSITION    = 'APPEND',
#else
     .           ACCESS      = 'APPEND',
#endif
#endif
     .           STATUS      = 'UNKNOWN',
     .           ERR         = 5100 )
	   WRITE (risc_buff, 3000) 'binary', fname
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0)
	ELSE
	   OPEN( UNIT        = default_list_lun,
     .           FILE        = fname,
     .           FORM        = 'FORMATTED',
#ifdef NO_ACCESS_APPEND
#else
#ifdef F90_OPEN_FILE_APPEND
     .		 POSITION    = 'APPEND',
#else
     .           ACCESS      = 'APPEND',
#endif
#endif
     .           STATUS      = 'UNKNOWN',
     .           ERR         = 5100 )
	   WRITE (risc_buff, 3000) 'ASCII', fname
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0)
	ENDIF
 3000	FORMAT (' Laplacian/spline gridded output is in ',A,
     .          ' file ',A)
	risc_buff = ' '
	WRITE (risc_buff, 3100) npts, nx, ny
	CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0)
 3100	FORMAT (' Input 3-tuples:',I5,'.   Output grid:',I4,' x'
     .           ,I4,'.')

* save the results
* Note: binary results are written one value per record - not the most
*	efficient but so that /COLUMNS= need not be known to read the
*	file back in
	IF ( fmt .EQ. ' ' ) THEN
	   DO 500 j = 1, ny
	   DO 500 i = 1, nx
 500	   WRITE (default_list_lun,ERR=5100) result(i,j)
	ELSE
	   DO 510 j = 1, ny
#ifdef DEC_FORTRAN
* on DEC compilers a field too small for the value generates an ERR= branch
	      WRITE (default_list_lun,fmt,IOSTAT=ios) (result(i,j),i=1,nx)
              IF ( ios.NE.0 .AND. ios.NE.63 ) GOTO 5100  ! 63 = fmt conv err
 510	   CONTINUE
#else
 510	   WRITE (default_list_lun,fmt,ERR=5100) (result(i,j),i=1,nx)
#endif
	ENDIF

 4900	CLOSE( UNIT=default_list_lun, ERR = 5000 )

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_erreq,status,fname,*4900 )

	END

