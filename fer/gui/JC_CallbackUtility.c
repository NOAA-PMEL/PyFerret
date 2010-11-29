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
 * JC_CallbackUtility.c
 *
 * Jonathan Callahan
 * Feb 19th 1996
 *
 * This file contains utility callback functions for common tasks.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 */


/* .................... Includes .................... */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <ctype.h>
#include <stdio.h>
#include <Xm/Xm.h>
 

/* .................... External Declarations .................... */



/* .................... Internal Declarations .................... */

void JC_TextField_VerifyNumeric_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
/*
 * This code is modified from the example on p. 500 of the O'Reilly X Series vol. Six A.
 */

     XmTextVerifyCallbackStruct *cbs = (XmTextVerifyCallbackStruct *) call_data;
     int len = 0;
     
/*
 * Don't verify if a convience routine is setting the TextField.
 *
 * Always allow the user to backspace.
 *
 * If the entry is not a digit or period, don't allow that entry.
 */

     if ( cbs->event == NULL )
	  return;

     if ( cbs->startPos < cbs->currInsert )
	  return;

     for ( len=0; len<cbs->text->length; len++ ) {
	  if ( !isdigit(cbs->text->ptr[len]) && cbs->text->ptr[len] != '.' ) {
	       cbs->doit = FALSE;
	       return;
	  }
     }

}


void JC_Message_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     /*
      * code fragment copied from page 128 of O'Reilly "X Series" vol. Six A.
      */
     Widget dialog=(Widget) NULL;
     Arg arg[5]={ NULL, };
     XmString t=XmStringCreateLocalized((char *)client_data);
     int n=0;

     XtSetArg (arg[n], XmNmessageString, t); n++;
     dialog = (Widget) XmCreateInformationDialog(wid, "message", arg, n);
     XtUnmanageChild( (Widget) XmMessageBoxGetChild(dialog, XmDIALOG_CANCEL_BUTTON));
     XtUnmanageChild( (Widget) XmMessageBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));

     XmStringFree(t);
     XtManageChild(dialog);
}
