      subroutine free_time
c
c  this routine calls atc_update every fourth time to refresh a GKS graphics
c  window.
c
c  kob 5/92
* 4/94 *sh* change "atc_update" to "PROCESS_X_EVENTS" to accomodate
*	Solaris/XGKS update needs (failure to redraw window)
*
* IBM PORT 14apr94 *kob* : IBM wouldn't allow c-like logical statment
*		 	  had to use ".NEQV" instead.
* IBM PORT 2jun94 *kob*  : Above change incorrect.  A mistake was made, the
*			   comparison should be done with .NE.		
      integer count
!      static count
      save count
      data count /0/


      if (mod(count+1,8) .NE. 0) then
         count = count +1
      else
c  call atc_update to refresh screen
!         call atc_update            ! 4/94
         call PROCESS_X_EVENTS       ! 4/94   
         count = 0
      endif

      return 

      end
