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
 * JC_Synchronize.c
 *
 * Jonathan Callahan
 * March 13th 1996
 *
 * This file contains functions for synchronizing the GUI with Ferret.
 *
 * The following global syncrhonization lists are maintained:
 *
 *
 * (LIST *)  dsets     global    windows     grids    viewports
 *             |        vars        |          |          |
 *             |         |          |          |          |
 *             |         V          V          V          V
 *             |
 *         <___|_______________________________________>
 *                 |              |                |
 *               dset1          dset2            dset3
 *             ____|____      ____|____        ____|____
 * struct {   |    |    |    |    |    |      |    |    |
 * (LIST *)  var  dvar cvar  |    |    |      |    |    |
 * }          |    |    |    |    |    |      |    |    |
 *            V    V    V    V    V    V      V    V    V
 * 
 *
 * This structure should help in the following tasks:
 *
 *  -  synchronizing the GUI with Ferret
 *  -  checking user-specified names against other names in a dataset to prevent duplication
 *  -  deleting an entire dataset
 * 
 * 96.12.12 Removed #ifdef FULL_GUI_VERSION 
 */


/* .................... Includes .................... */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>

#include <Xm/Xm.h>
#include "UxXt.h"

#include "ferret_structures.h"
#include "ferret.h"
#include "ferret_shared_buffer.h"


/* .................... External Declarations .................... */

extern LIST *GLOBAL_DatasetList;

extern int ferret_query(int query, smPtr sBuffer, char *tag,
		 char *arg1, char *arg2, char *arg3, char *arg4 );
extern char *CollectToReturn(char *targetStr, char *subStr);
extern void DisplayMacroBuffer(void);
extern void ShowCursorPos(void);
extern void ferret_error(smPtr iBuffer);
extern void ferret_pause(void);

extern char *macroBuffer;
extern int gMacroIsRecording, gMMIsOpen, gOlineCount, mmPos;
extern float *memory;
extern swidget FerretMainWd, ListManager;


/* .................... Internal Declarations .................... */

void ferret_command(char *cmdText, int cmdMode);
int  GUIFerretDispatch(char *inCmd);

#define DONT_UPDATE_MM 3


/* .................... Function Definitions .................... */

void ferret_command( char *cmd, int mode)
{
  int GUIFerretDispatch_return_status=1;
  int JC_II_ND=0;

  GUIFerretDispatch_return_status = GUIFerretDispatch(cmd);
     
  /*
   * UPDATE_COMMAND_WIDGET and IGNORE_COMMAND_WIDGET are not currently used.
   */

  /*
   * If the macro manager needs updating:
   *    - Test whether the last command ended with a return
   *    - Append the command to the macro buffer.
   *    - If the macro display is open: update it.
   *
   * If this was a "LIST" command:
   *    - Display the ListManager.
   */

  if ( GUIFerretDispatch_return_status == 0 ) {

    if ( gMacroIsRecording && mode != DONT_UPDATE_MM ) {
      if (macroBuffer[strlen(macroBuffer)-1] != '\n' && mmPos)
	strcat(macroBuffer, "\n");
      strcat(macroBuffer, cmd);
      strcat(macroBuffer, "\n");

      if (gMMIsOpen)
	DisplayMacroBuffer();
    }

    /* JC_TODO ... (ferret_command) Popping up the list manager should be done in JC_InterInterface. */
	  
    if (strstr("LIST", cmd)) {
      XtPopup(UxGetWidget(ListManager), no_grab);
      ShowCursorPos();
      gOlineCount = 0;
		
    }

  } else {
    /*
      fprintf(stderr, "WARNING in JC_Synchronize.c: ferret_command(): GUIFerretDispatch_return_status = %d\n",
      GUIFerretDispatch_return_status);
      */	  
  }

  switch( (JC_II_ND = JC_II_Synchronize(NULL)) ) {

  case -2:
    fprintf(stderr, "ferret_command: JC_II_Synchronize() returned FATAL_ERR\n");
    exit(1);
    break;

  case -1:
    /* this is a NOT_FOUND_ERR which is normal when the gui first starts up */
    break;

  case 0:
    break;

  default:
    fprintf(stderr, "ferret_command: JC_II_Synchronize() returned %d\n", JC_II_ND);
    exit(1);
    break;
		
  }
     
}


int GUIFerretDispatch( char *inCmd)
{
  char command[MAX_COMMAND_LENGTH]="";
  Boolean Ferret_is_working=TRUE;
  int max_mem_blks=PMAX_MEM_BLKS, mem_blk_size=0,  old_mem_blk_size=0;
  int mem_size=PMEM_BLK_SIZE * PMAX_MEM_BLKS;
  Status err=0;

  strcpy(command, inCmd);
     
  while (Ferret_is_working) {

    /* JC_TODO ... (GUIFerretDispatch) Code doesn't handle returns when Ferret is still working. */
  
    /*
     * - Send the command to Ferret.
     * - Test whether Ferret is still working on a command stack.
     * - If Ferret is still working:
     *      Switch on the FRTN_ACTION flag in the shared memory buffer.
     * - Else if Ferret is done:
     *      Switch on the FRTN_ACTION flag in the shared memory buffer.
     */
    ferret_dispatch_c(memory, command, sBuffer);
	  
    if (sBuffer->flags[FRTN_CONTROL] == FCTRL_IN_FERRET)
      strcpy(command, "");
    else
      Ferret_is_working = False;
	  
    if ( Ferret_is_working ) {
	       
      switch (sBuffer->flags[FRTN_ACTION]) {
	       
      case FACTN_NO_ACTION:
	continue;
	break;

      case FACTN_DISPLAY_WARNING:
      case FACTN_DISPLAY_ERROR:
      case FACTN_DISPLAY_TEXT:
      case FACTN_SYNCH_SET_DATA:
      case FACTN_SYNCH_LET:
      case FACTN_SYNCH_WINDOW:
      case FACTN_MEM_RECONFIGURE:
      case FACTN_EXIT:
	;/* I need to handle these somehow */
      break;
		    
      case FACTN_PAUSE:
	ferret_pause();
	break;
		    
      default:
	fprintf(stderr, "ERROR in JC_Synchronize: GUIFerretDispatch(): FRTN_ACTION = %d\n",
		sBuffer->flags[FRTN_ACTION]);
	break;
	       
      }
	       
    } else /* Ferret is done working */ {
	       
      switch (sBuffer->flags[FRTN_ACTION]) {
	       
      case FACTN_NO_ACTION:
	continue;
	break;

      case FACTN_DISPLAY_WARNING:
      case FACTN_DISPLAY_TEXT:
	;
      break;

      case FACTN_DISPLAY_ERROR:
	ferret_error(sBuffer);
	break;
		    
      case FACTN_PAUSE:
	;
      break;
	       
      case FACTN_SYNCH_SET_DATA:
	;
      break;
		    
      case FACTN_SYNCH_LET:
	fprintf(stderr, "UNFINISHED CODE: Defined Variable synchronization has not been completed yet.\n");
	break;
	       
      case FACTN_SYNCH_WINDOW:
	if (sBuffer->flags[FRTN_IDATA2] > 0) {
	  /* window created--install close window handler*/
	  Display **dpy;
	  GC gc;
	  Atom wmDeleteWindow, wmProtocols, atoms[2];
	  Widget shell;
	  XTextProperty wmName;
	  char *sName="1";
	  Window outWin;
	  int outWinNum;

	  /* get the widget for this particular window */
	  outWinNum = sBuffer->flags[FRTN_IDATA1];
	  gescinqxattr(outWinNum, &dpy, &outWin, &gc);

			 /* add the close window code 
			    XStringListToTextProperty(&sName, 1, &wmName);
			    XSetWMName(XtDisplay(FerretMainWd), outWin, &wmName); */
	  wmDeleteWindow = XInternAtom(
				       XtDisplay(FerretMainWd),
				       "WM_DELETE_WINDOW",
				       FALSE);
				
	  wmProtocols = XInternAtom(
				    XtDisplay(FerretMainWd),
				    "WM_PROTOCOLS",
				    FALSE);

	  atoms[0] = wmProtocols;
	  atoms[1] = wmDeleteWindow;
	  err = XSetWMProtocols(XtDisplay(FerretMainWd), outWin, atoms, 2);
	}
	break;
		    
      case FACTN_MEM_RECONFIGURE:
	old_mem_blk_size = mem_blk_size;
	mem_size = sBuffer->flags[FRTN_IDATA1];
	mem_blk_size = mem_size / max_mem_blks;
		    
	free((void *)memory);
	memory = (float *) malloc(mem_size*sizeof(float));
	if (memory == 0) {
	  printf("Unable to allocate %d Mwords of memory.\n",mem_size/1.E6 );
	  mem_blk_size = old_mem_blk_size;
	  mem_size = mem_blk_size * max_mem_blks;
	  memory = (float *) malloc(mem_size*sizeof(float));
	  if (memory == 0) {
	    printf("Unable to reallocate previous memory of %d Mwords.\n",mem_size/1.E6 );
	    exit(0);
	  } 
	  else
	    printf("Restoring previous memory of %f Mwords.\n",mem_size/1.E6 );
	}
#ifdef NO_ENTRY_NAME_UNDERSCORES
	init_memory( &mem_blk_size, &max_mem_blks );
#else
	init_memory_( &mem_blk_size, &max_mem_blks );
#endif
	break;
	       
      case FACTN_EXIT:
	printf("Exit from FERRET requested\n");
	/*   FORTRAN(finalize)();*/
	exit(0);
	break;
	       
      default:
	fprintf(stderr, "ERROR in JC_Synchronize: GUIFerretDispatch(): FRTN_ACTION = %d\n",
		sBuffer->flags[FRTN_ACTION]);
	break;
	       
      } /* switch */
	       
    } /* if */

  } /* while */
     

  if ( sBuffer->flags[FRTN_ACTION] == FACTN_DISPLAY_ERROR )
    return sBuffer->flags[FRTN_ACTION];
  else
    return 0;

}

