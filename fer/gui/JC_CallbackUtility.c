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
