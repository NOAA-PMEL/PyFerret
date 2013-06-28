      SUBROUTINE NEW_GKS_WINDOW (ws_id,xname,window_title,status)

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
*     Modifies connection id string, opens GKS workstation.  Permits multiple 
*     windows on workstation screen.  Modified from GKS-3D FORTRAN Binding
*     Manual example 19-1.  For VMS simply open the workstation.
*     J Davison 5.13.91
*     *jd* 3.26.92  Modified to accommodate ATC GKS as well as DEC
*     *jd* 12.21.93 Modified to accommodate xgks
*  V600 *acm* 9/05 append window_title to window name

      IMPLICIT NONE

* Calling arguments
      integer         ws_id,status
      character*(*)   xname, window_title

* Internal declarations
      integer         TM_LENSTR1, nlen
      character*2     kwsid
      character*88    window_name

      include         'gkspar.inc' 
      
      write (unit=kwsid,fmt='(i2)') ws_id

      window_name = 'FERRET_'//kwsid(2:2)
      nlen = TM_LENSTR1(window_title)
      IF (nlen .GT. 1 .OR. window_title .NE. '_')
     .    window_name = 'FERRET_'//kwsid(2:2)//'_'//window_title
      call FGD_GESSPN(window_name)
      call FGD_GOPWK(ws_id,6,4)

*     SET DEFERRAL STATE TO GENERATE OUTPUT AS SOON AS POSSIBLE
      call FGD_GSDS(ws_id,gasap,gsuppd)
      status = 3   ! Meaningless but may be used later
      return 
      end
      

