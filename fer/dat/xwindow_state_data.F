         BLOCK DATA XWINDOW_STATE_DATA
* *kob* New routine for Linux/f90 port

#ifdef unix
        include 'ferret_cmn/xwindow_state.cmn'
#else
        INCLUDE 'FERRET_CMN:XWINDOW_STATE.CMN'
#endif

*               ==> COMMON XWINDOW_STATE has been created **temporarily**
*                   with the intention that it will be expanded in future
*                   so overlays can be placed on **previously** drawn windows


* pre-initialize to [0,1] limits just to avoid misbehavior from non-init
         DATA  haxlo, haxhi, vaxlo, vaxhi/0.D0,1.D0,0.D0,1.D0/


	END
