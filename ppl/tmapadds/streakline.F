        SUBROUTINE streakline (u, v, xf, yf)
	REAL u, v, xf, yf

C**
C**    @(#)streakline.F	    9/19/00
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
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
*  ACM  implementing algorighm from grads -- gxstrm.c
*       draw streaklines 

C	ARRAY X HOLD 2 GRIDS, U(X) IN THE FIRST
C	NX BY NY GRID AND V(Y) IN THE SECOND
C
	DIMENSION U(NX,NY),V(NX,NY)
C
C

#ifdef unix
	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
#endif

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'plt_inc.decl'
	include 'PLT.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'cont_inc.decl'
	include 'CONT.INC'
	include 'zgrid_inc.decl'
	include 'ZGRID.INC'
	include 'zgridd_inc.decl'
	include 'ZGRIDD.INC'
	include 'vector_inc.decl'
	include 'VECTOR.INC'
#ifdef NO_PASSED_CONCAT	
	include 'xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'tmap_pplv11inc:tmap_axis.inc'
	INCLUDE 'tmap_pplv11inc:tmap_plt.inc'
	INCLUDE 'tmap_pplv11inc:tmap_miss.inc'
	INCLUDE 'tmap_pplv11inc:tmap_hd.inc'
	INCLUDE 'tmap_pplv11inc:tmap_cont.inc'
	INCLUDE 'tmap_pplv11inc:tmap_zgrid.inc'
	INCLUDE 'tmap_pplv11inc:tmap_zgridd.inc'
	INCLUDE 'tmap_pplv11inc:tmap_vector.inc'

#endif


*******************************************************************************



C  From GrADS code, gxstrm.c

C void gxstrm (float *u, float *v, float *c, int is, int js,
C   float uund, float vund, float cund, int flag, float *shdlvs,
C   int *shdcls, int shdcnt, int den) 

      
      REAL    memory( mem_blk_size, max_mem_blks )
      INTEGER  mblk1, nblks1, status

C  from gxstrm.c Figure out the interval for the flag grid

      integer siz, i, iscl, iss, jss
      i = NX
      if (NY .GT. i) i = NY
      iscl = 200/i
      iscl = iscl + density - 5
      if (iscl .LT. 1) iscl=1
      if (iscl .GT. 10) iscl=10
 
C   Allocate memory for the flag grid

      iss = NX*iscl
      jss = NY*iscl
      siz = iss*jss

      CALL GET_MEMORY( 4*siz, mblk1,nblks1,status )
      IF ( status .NE. ferr_ok ) GOTO 5000

      CALL streakline_sub (u, v, memory(1,mblk1), siz, density, xf, yf)

* Cleanup
      CALL FREE_MEMORY( mblk1, nblks1 )

* error exits
 5000 RETURN

      END
