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


