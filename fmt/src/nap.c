/*
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
*/



/* routine to put ferret asleep.

   modifed 8/95 for solaris...default routine was causing major
   delay's while typing and cutting and pasting

*/


#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>


#define setvec(vec,a) \
            vec.sv_handler = a; \
            vec.sv_mask    = vec.sv_flags = 0


static int ringring;


nap(n)
unsigned n;

{

#ifdef sgi
  sginap(10);
#elif solaris
  fd_set fdset;
  struct timeval time;
  FD_ZERO(&fdset);
  time.tv_sec = n/60;
  time.tv_usec =(n/3.9)*1000000/60;  
  select(0, 0, 0, 0, &time);
#else
  void napx();
  long omask;
  struct sigvec vec,ovec;
  struct itimerval itv, oitv;
  register struct itimerval *itp = &itv;


  if ( n == 0)
    return;

  timerclear (&itp->it_interval);
  timerclear (&itp->it_value);

  if (setitimer(ITIMER_REAL, itp, &oitv) < 0)
    return;

  setvec (ovec,SIG_DFL);
  omask = sigblock(sigmask(SIGALRM));

  itp->it_value.tv_sec = n/60;
  itp->it_value.tv_usec = (n%60)*1000000/60;

  if (timerisset(&oitv.it_value)) {
    if (oitv.it_value.tv_sec >= itp->it_value.tv_sec) {
      if (oitv.it_value.tv_sec == itp->it_value.tv_sec &&
	  oitv.it_value.tv_usec > itp->it_value.tv_usec)
	oitv.it_value.tv_usec -= itp->it_value.tv_usec;
        oitv.it_value.tv_sec -= itp->it_value.tv_sec;
    }
    else {
      itp->it_value = oitv.it_value;

      oitv.it_value.tv_sec = 1;
      oitv.it_value.tv_usec = 0;
    }
  }

  setvec(vec,napx);
  ringring = 0;
  
  sigvec(SIGALRM,&vec,&ovec);
  setitimer(ITIMER_REAL, itp, (struct itimerval *)0);

  while (!ringring)
    sigpause(omask &~ sigmask(SIGALRM));

  sigvec(SIGALRM, &ovec, (struct sigvec *) 0);
  setitimer(ITIMER_REAL, &oitv, (struct itimerval *) 0);
  sigsetmask(omask);
#endif

}


#ifndef sgi
static void
napx()
{
  ringring = 1;
}

#endif


