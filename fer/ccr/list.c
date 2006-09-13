/* list.c -- a generic list package
 * 
 * Last edited: Tue Jul 28 15:37:24 1992 by bcs (Bradley C. Spatz) on wasp
 *
 * Copyright (C) 1992, Bradley C. Spatz, bcs@ufl.edu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *
 * We define the following routines here:
 *
 *    LIST *list_init()
 *    LIST *list_mvprev(list)
 *    LIST *list_mvnext(list)
 *    char *list_insert_before(list, data, bytes)
 *    char *list_insert_after(list, data, bytes)
 *    char *list_remove_front(list)
 *    char *list_remove_curr(list)
 *    char *list_remove_rear(list)
 *    int list_traverse(list, data, func, opts);
 *    void list_free(list, dealloc)
 *
 * and the following here or in list.h if USE_MACROS defined for compilation.
 *
 *    char *list_front(list)
 *    char *list_curr(list)
 *    char *list_rear(list)
 *    LIST *list_mvfront(list)
 *    LIST *list_mvrear(list)
 *    int list_size(list)
 *    int list_empty(list)
 *
 * for
 *
 *    LIST *list;
 *    char *data;
 *    int bytes;
 *    int func(data, curr)
 *       char *data;
 *       char *curr;
 *    void dealloc(data)
 *       char *data;
 */

static char brag[] = "$$Version: list-2.1 Copyright (C) 1992 Bradley C. Spatz";

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>
#include "list.h"

/*char *malloc();*/


LIST *list_init()
{
   LIST *list;

   /* Allocate, initialize, and return a new list. */
   list = (LIST *) malloc(sizeof(LIST));
   list->size = 0;
   list->front = NULL;;
   list->rear = NULL;;
   list->curr = NULL;;
   return(list);
}


LIST *list_mvprev(list)
LIST *list;
{
   /* Move to the previous link, if possible.  Note that the following
    * compound conditional expression *requires* a short-circuit evaluation.
    */
   if ((list->curr != NULL) && (list->curr->prev != NULL)) {
      list->curr = list->curr->prev;
      return(list);
   }
   else
      return(NULL);
}


LIST *list_mvnext(list)
LIST *list;
{
   /* Move to the next link, if possible.  Note that the following
    * compound conditional expression *requires* a short-circuit evaluation.
    */
   if ((list->curr != NULL) && (list->curr->next != NULL)) {
      list->curr = list->curr->next;
      return(list);
   }
   else
      return(NULL);
}


/* The following are definitions of these routines as functions.  We can
 * force thse to be implemented as macros by compiling with -DUSE_MACROS.
 * The macros are defined in the header file(s).  The macros afford better
 * performance, and if users know the routines are implemented as such,
 * they can always wrap their own functions aroud the macros if they need
 * function semantics (i.e. using the routines as pointers, as in passing
 * the routines as parameters to other functions.
 */
#ifndef USE_MACROS
LIST *list_mvfront(list)
LIST *list;
{
   /* Move to the front of the list.*/
   list->curr = list->front;
   return(list);
}


LIST *list_mvrear(list)
LIST *list;
{
   /* Move to the front of the list.*/
   list->curr = list->rear;
   return(list);
}


int list_empty(list)
LIST *list;
{
   /* Return 1 if the list is empty.  0 otherwise. */
   return((list->front == NULL) ? TRUE : FALSE);
}


char *list_front(list)
LIST *list;
{
   return((list->front == NULL) ? NULL : (list->front->data));
}


char *list_curr(list)
LIST *list;
{
   return((list->curr == NULL) ? NULL : (list->curr->data));
}


char *list_rear(list)
LIST *list;
{
   return((list->rear == NULL) ? NULL : (list->rear->data));
}


int list_size(list)
LIST *list;
{
   return(list->size);
}
#endif


static LIST_ELEMENT *list_create_element(data, bytes)
char *data;
int bytes;
{
   LIST_ELEMENT *new;

   /* Allocate storage for the new node and its data.  Return NULL if
    * unable to allocate.
    */
   new = (LIST_ELEMENT *) malloc(sizeof(LIST_ELEMENT));
   if (new == NULL) {
      return(NULL);
   }

   /* Allocate storage for the data only if requested; i.e. if bytes > 0.
    * Then either copy the data or just the reference into the node.
    */
   if (bytes > 0) {
      new->data = (char *) malloc(bytes);
      if (new->data == NULL) {
	 return(NULL);
      }
      (void) memcpy(new->data, data, bytes);
   }
   else {
      new->data = (char *) data;
   }

   return(new);
}


char *list_insert_before(list, data, bytes)
LIST *list;
char *data;
int bytes;
{
   LIST_ELEMENT *new;

   /* Allocate storage for the new element and its data.*/
   new = list_create_element(data, bytes);
   if (new == NULL)
      return(NULL);

   /* Now insert the element before the current, considering the cases:
    *    1) List is empty
    *    2) Inserting at front
    *    3) Otherwise
    * We handle them directly, in order.
    */
   if (list->front == NULL) {
      /* The list is empty.  Easy. */
      new->prev = new->next = NULL;
      list->front = list->rear = list->curr = new;
   }
   else if (list->curr->prev == NULL) {
      /* Inserting at the front. */
      new->prev = NULL;
      new->next = list->curr;
      list->curr->prev = new;
      list->front = new;
   }
   else {
      /* Otherwise. */
      new->prev = list->curr->prev;
      list->curr->prev->next = new;
      new->next = list->curr;
      list->curr->prev = new;
   }

   list->curr = new;
   list->size++;
   return(new->data);
}


char *list_insert_after(list, data, bytes)
LIST *list;
char *data;
int bytes;
{
   LIST_ELEMENT *new;

   /* Allocate storage for the new element and its data.*/
   new = list_create_element(data, bytes);
   if (new == NULL)
      return(NULL);

   /* Now insert the element after the current, considering the cases:
    *    1) List is empty
    *    2) Inserting at rear
    *    3) Otherwise
    * We handle them directly, in order.
    */
   if (list->front == NULL) {
      /* The list is empty.  Easy. */
      new->prev = new->next = NULL;
      list->front = list->rear = list->curr = new;
   }
   else if (list->curr->next == NULL) {
      /* Inserting at the rear. */
      new->next = NULL;
      new->prev = list->curr;
      list->curr->next = new;
      list->rear = new;
   }
   else {
      /* Otherwise. */
      new->next = list->curr->next;
      new->next->prev = new;
      new->prev = list->curr;
      list->curr->next = new;
   }

   list->curr = new;
   list->size++;
   return(new->data);
}


static char *list_remove_single(list)
LIST *list;
{
   char *data;

   /* The list has one element.  Easy. */
   data = list->curr->data;
   free(list->curr);
   list->front = list->rear = list->curr = NULL;
   list->size--;
   return (data);
}


char *list_remove_front(list)
LIST *list;
{
   LIST_ELEMENT *temp;
   char *data;

   /* Removing and return front element, or NULL if empty.  If curr
    * is the front, then curr becomes the next element.
    */
   if (list->front == NULL) {
      /* List is empty.  Easy. */
      return(NULL);
   }
   else if (list->front == list->rear) {
      /* List has only one element.  Easy. */
      data = list_remove_single(list);
   }
   else {
      /* List has more than one element.  Make sure to check if curr
       * points to the front.
       */
      data = list->front->data;
      list->front->next->prev = NULL;
      temp = list->front;
      list->front = temp->next;
      if (list->curr == temp)
	 list->curr = temp->next;
      free(temp);
      list->size--;
   }

   return(data);
}


char *list_remove_rear(list)
LIST *list;
{
   LIST_ELEMENT *temp;
   char *data;

   /* Removing and return rear element, or NULL if empty.  If curr
    * is the rear, then curr becomes the previous element.
    */
   if (list->front == NULL) {
      /* List is empty.  Easy. */
      return(NULL);
   }
   else if (list->front == list->rear) {
      /* List has only one element.  Easy. */
      data = list_remove_single(list);
   }
   else {
      /* List has more than one element.  Make sure to check if curr
       * points to the rear.
       */
      data = list->rear->data;
      list->rear->prev->next = NULL;
      temp = list->rear;
      list->rear = temp->prev;
      if (list->curr == temp)
	 list->curr = temp->prev;
      free(temp);
      list->size--;
   }

   return(data);
}


char *list_remove_curr(list)
LIST *list;
{
   LIST_ELEMENT *temp;
   char *data;

   /* Remove the current element, returning a pointer to the data, or
    * NULL if the list is empty.  Set curr to the next element unless
    * curr is at the rear, in which case curr becomes the previous
    * element.
    */
   if (list->front == NULL) {
      /* The list is empty.  Easy. */
      return(NULL);
   }
   else if (list->front == list->rear) {
      /* The list has one element.  Easy. */
      data = list_remove_single(list);
   }
   else if (list->curr == list->front) {
      /* Removing front element.  Easy. */
      data = list_remove_front(list);
   }
   else if (list->curr == list->rear) {
      /* Removing from the rear.  Easy.*/
      data = list_remove_rear(list);
   }
   else {
      /* Otherwise.  Must be inside a 3-element or larger list. */
      data = list->curr->data;
      temp = list->curr;
      temp->next->prev = temp->prev;
      temp->prev->next = temp->next;
      list->curr = temp->next;
      free(temp);
      list->size--;
   }

   return(data);
}


int list_traverse(list, data, func, opts)
LIST *list;
char *data;
int (*func)();
int opts;
{
   LIST_ELEMENT *lp;
   int status, rc;

   /* Traverse the list according to opts, calling func at each element,
    * until func returns 0 or the extent of the list is reached.  We return
    * 0 if the list is empty, 2 if we tried to go beyond the extent of the
    * list, and 1 otherwise.  We may or may not affect the current element
    * pointer.
    */
   if (list->front == NULL)
      return(LIST_EMPTY);
   
   /* Decide where to start. */
   if ((opts & LIST_CURR) == LIST_CURR) {
      lp = list->curr;
   }
   else if ((opts & LIST_REAR) == LIST_REAR) {
      lp = list->rear;
   }
   else {
      lp = list->front;
   }
   
   /* Now decide if to update the current element pointer. */
   if ((opts & LIST_ALTR) == LIST_ALTR)
      list->curr = lp;

   /* Now go until 0 is returned or we hit the extent of the list. */
   rc = LIST_OK;
   status = TRUE;
   while(status) {
      status = (*func)(data, lp->data);
      if (status) {
	 if ((((opts & LIST_BACK) == LIST_BACK) ? (lp->prev) : (lp->next))
	     == NULL) {
	    /* Tried to go beyond extent of list. */
	    status = FALSE;
	    rc = LIST_EXTENT;
	 }
	 else {
	    /* Decide where to go next. */
	    lp = (((opts & LIST_BACK) == LIST_BACK) ? (lp->prev) : (lp->next));

	    /* Now decide if to update the current element pointer. */
	    if ((opts & LIST_ALTR) == LIST_ALTR)
	       list->curr = lp;
	 }
      }
   }

   return(rc);
}


void list_free(list, dealloc)
LIST *list;
void (*dealloc)();
{
   char *data;

   /* Move to the front of the list.  Start removing elements from the
    * front.  Free up the data element by either applying the user-supplied
    * deallocation routine or free(3).  When we've gone through all the
    * elements, free the list descriptor.
    */
   list_mvfront(list);
   while (! list_empty(list)) {
      data = list_remove_front(list);
      /* Apply either no deallocation function to each node, our own, or
       * a user-supplied version.
       */
      if ((int) dealloc != LIST_NODEALLOC) {
	 if ((int) dealloc == LIST_DEALLOC) {
	    free(data);
	 }
	 else {
	    (*dealloc)(data);
	 }
      }
   }

   free(list);
}
