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

/*
 * Init gui dependent code
 * JS 3.99
 */

#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Xlib.h>
#include "ferret.h"
/*---------------------------------------------------- 
 * UxXt.h needs to be included only when compiling a 
 * stand-alone application. 
 *---------------------------------------------------*/
#define __globalDefs
#include "ferret_fortran.h"
#include "ferret_structures.h"
#include "ferret_shared_buffer.h"
#include "JC_Utility.h" /* for windows[] and window_count in the SGI_POPUPS section */
#undef __globalDefs
#ifndef DESIGN_TIME
#include "UxXt.h"
#endif /* DESIGN_TIME */

XtAppContext	UxAppContext;
Widget		UxTopLevel;
Display		*UxDisplay;

int		UxScreen;
char init_command[128];
extern void SetInitialState(void);

/*----------------------------------------------
 * Insert application global declarations here
 *---------------------------------------------*/
/*
 * BEGIN of "FerretMain_mycode_1.c"
 *
 * Here are the contents of the file "FerretMain_mycode_1.c"
 * which was previously included as a separate file.
 */

/* .................... Function Definitions .................... */

void help_text(void);
static void TimeOutCB(XtPointer cd, XtIntervalId *id);

/* .................... Function Declarations .................... */

static void TimeOutCB(XtPointer cd, XtIntervalId *id)
{
  XtAppAddTimeOut((XtAppContext)cd, (unsigned long)2000, TimeOutCB, cd);
  
  FORTRAN(xgks_x_events)();

}

static void do_gui_init() {

  /*
   * Make sure the FERRET environment is set up and ready by looking for $FER_DIR.
   */
  if (getenv("FER_DIR") == NULL){
    printf("\n  The FERRET environment has not been properly set up.\n");
    printf("  (The environment variable FER_DIR is not defined)\n");
    printf("\n  Have you executed \"source your_system_path/ferret_paths\" ?\n");
    exit(1);
  }
}

#ifndef LINK_GUI_AS_MAIN
int gui_init() {
  return 0;
}

static float *mem;
				
float **gui_get_memory() {
  return &mem;
}
void gui_run(int *argc, char **argv){
}

#else

int gui_init() {
  int rval = 1;
  do_gui_init();
  return rval;
}

				/* Global variable in ferret_shared.h */
float **gui_get_memory() {
  return &memory;
}

void gui_run(int *argc, char **argv){


  char tText[128], quote='"';
  extern void SetInitialState(void);

  /*-----------------------------------------------------------
   * Declarations.
   * The default identifier - mainIface will only be declared 
   * if the interface function is global and of type swidget.
   * To change the identifier to a different name, modify the
   * string mainIface in the file "xtmain.dat". If "mainIface"
   * is declared, it will be used below where the return value
   * of  PJ_INTERFACE_FUNCTION_CALL will be assigned to it.
   *----------------------------------------------------------*/ 

  Widget mainIface;

  /*---------------------------------
   * Interface function declaration
   *--------------------------------*/	

  Widget  create_FerretMainWd(swidget UxParent);
  swidget UxParent = NULL;


  /*------------------------------------------------------
   * Declare the fallback resources (added by JC 9.13.96)
   *-----------------------------------------------------*/

#include "fallback_resources.c"

#ifdef XOPEN_CATALOG
  if (XSupportsLocale()) {
    XtSetLanguageProc(NULL,(XtLanguageProc)NULL,NULL);
  }
#endif

  /*
   * ==>> place FERRET under GUI control here  <<== 
   */
  FORTRAN(mode_gui_on)();

  /*UxTopLevel = XtAppInitialize(&UxAppContext, "Ferret",
      NULL, 0, &argc, argv, NULL, NULL, 0);*/

  UxTopLevel = XtVaAppInitialize(&UxAppContext, "Ferret",
				 NULL, 0, argc, argv, fallbacks, NULL);

  UxDisplay = XtDisplay(UxTopLevel);
  UxScreen = XDefaultScreen(UxDisplay);

  /*
     * We set the geometry of UxTopLevel so that dialogShells
     * that are parented on it will get centered on the screen
     * (if defaultPosition is true).
     */

  XtVaSetValues(UxTopLevel,
		XtNx, 0,
		XtNy, 0,
		XtNwidth, DisplayWidth(UxDisplay, UxScreen),
		XtNheight, DisplayHeight(UxDisplay, UxScreen),
		NULL);

  /*-------------------------------------------------------
     * Insert initialization code for your application here 
     *------------------------------------------------------*/
	
    /*----------------------------------------------------------------
     * Create and popup the first window of the interface.  The 	 
     * return value can be used in the popdown or destroy functions.
     * The Widget return value of  PJ_INTERFACE_FUNCTION_CALL will 
     * be assigned to "mainIface" from  PJ_INTERFACE_RETVAL_TYPE. 
     *---------------------------------------------------------------*/

  mainIface = create_FerretMainWd(UxParent);

  SetInitialState();
  UxPopupInterface(mainIface, no_grab);

#ifdef X_REFRESH
  XtAppAddTimeOut(UxAppContext, 2000, TimeOutCB, UxAppContext);
#endif

#ifdef SGI_POPUPS
  /* NB__ UxTopLevel is declared external in UxXt.h */
  windows[window_count++] = XtWindow(UxTopLevel);
  XSetWMColormapWindows(XtDisplay(UxTopLevel), XtWindow(UxTopLevel),
			windows, window_count);
#endif /* SGI_POPUPS */

  /*-----------------------
     * Enter the event loop 
     *----------------------*/
  {

    XEvent event;

    for (;;)
      {
	XtAppNextEvent(UxAppContext, &event);

	if (event.type == ClientMessage) {
	  ;
	}
	switch (event.type)
	  {

	    /*---------------------------------------------------
	     * Insert code here to handle any events that you do
	     * not wish to be handled by the interface.
	     *---------------------------------------------------*/
	  default:
	    XtDispatchEvent(&event);
	    break;
	  }
      }

  } /* event loop block */
}
#endif /* LINK_GUI_AS_MAIN */
