/* This file written by C Brady, University of Warwick (2019)
 * The contents of this file are placed into the public domain
 * under the Creative Commons CC0 license
 * https://creativecommons.org/publicdomain/zero/1.0/legalcode
 * As per this license this work is provided as-is with no warranty
 * made of any kind*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <errno.h>
#include "../include/kvp.h"

kvp_item *kvp_items = NULL;
int kvp_itemcount=0;
int kvp_itemslots=0;


/*Test for valid range*/
static long long range_check(char *value, KVP_ERROR_TYPE *err, long long min, long long max){
	char* after;
	long long val = strtoll(value,&after,10);
	if (after!=value+strlen(value)) {
		*err = KVP_BAD_VALUE;
		return 0;
	}

	if (val < min || val > max){
		*err = KVP_BAD_RANGE;
		return 0;
	}

	*err = KVP_OK;
	return val;
}

static long long range_check_u(char *value, KVP_ERROR_TYPE *err, unsigned long long min, unsigned long long max){

	char* after;
	unsigned long long val = strtoull(value,&after,10);
	if (after!=value+strlen(value)) {
		*err = KVP_BAD_VALUE;
		return 0;
	}

	if (val < min || val > max){
		*err = KVP_BAD_RANGE;
		return 0;
	}

	*err = KVP_OK;
	return val;

}

static void kvp_expand_items(){

	if(kvp_itemcount < kvp_itemslots) return;

	/*Grow array by doubling each time*/
	if (kvp_items){
		kvp_item * temp = (kvp_item*)malloc(sizeof(kvp_item) * kvp_itemslots * 2);
		memcpy(temp, kvp_items, sizeof(kvp_item) * kvp_itemcount);
		free(kvp_items);
		kvp_items = temp;
		kvp_itemslots *= 2;
	} else {
		kvp_itemslots = 1;
		kvp_items = (kvp_item*)malloc(sizeof(kvp_item));
	}
}

/*Internal storage functions*/
static KVP_ERROR_TYPE kvp_store_c(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(char*)(data)) = range_check(value, &err, CHAR_MIN, CHAR_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_uc(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(unsigned char*)(data)) = range_check_u(value, &err, 0, UCHAR_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_s(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(short*)(data)) = range_check(value, &err, SHRT_MIN, SHRT_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_us(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(unsigned short*)(data)) = range_check_u(value, &err, 0, USHRT_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_i(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(int*)(data)) = range_check(value, &err, INT_MIN, INT_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_ui(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(unsigned int*)(data)) = range_check_u(value, &err, 0, UINT_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_l(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(long*)(data)) = range_check(value, &err, LONG_MIN, LONG_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_ul(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(unsigned long*)(data)) = range_check_u(value, &err, 0, ULONG_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_ll(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(long long*)(data)) = range_check(value, &err, LLONG_MIN, LLONG_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_ull(char *key, char *value, void *data){
	KVP_ERROR_TYPE err;
	(*(unsigned long long*)(data)) = range_check_u(value, &err, 0, ULLONG_MAX);
	return err;
}

static KVP_ERROR_TYPE kvp_store_f(char *key, char *value, void *data){
	char *after;
	(*(float*)(data)) = strtof(value,&after);
	if (after!=value+strlen(value)) return KVP_BAD_TYPE;
	if (errno == ERANGE) return KVP_BAD_RANGE;
	return KVP_OK;
}

static KVP_ERROR_TYPE kvp_store_d(char *key, char *value, void *data){
	char *after;
	(*(double*)(data)) = strtod(value,&after);
	if (after!=value+strlen(value)) return KVP_BAD_TYPE;
	if (errno == ERANGE) return KVP_BAD_RANGE;
	return KVP_OK;
}

static KVP_ERROR_TYPE kvp_store_string(char *key, char *value, void *data){
	kvp_held_string *khs = (kvp_held_string *)data;
	strncpy(khs->string, value, khs->size);
	return KVP_OK;
}

/*End internal storage functions*/

/*Core of the registration system*/
static kvp_item* kvp_register_core(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data), void *data, unsigned char free){
	kvp_expand_items();
	strncpy(kvp_items[kvp_itemcount].name, key, KVP_MAX_KEY);
	kvp_items[kvp_itemcount].action = action;
	kvp_items[kvp_itemcount].data = data;
	kvp_items[kvp_itemcount].free = free;
	++kvp_itemcount;
	return &kvp_items[kvp_itemcount-1];
}

/*Store registration in most typical case*/ 
void kvp_register(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data), void *data){
	kvp_register_core(key, action, data, 0);
}

/*Store callback function with no tied data*/
void kvp_register_function(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data)){
	kvp_register_core(key, action, NULL, 0);
}

/*Store callback function tied data*/
void kvp_register_function_with_data(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data), void *data){
	kvp_register_core(key, action, data, 0);
}

/*Convenience functions to register storing to variables
 *This allows you to go straight from a key to a variable
 * without the host code having to do anything
 * */

/*Register char*/
void kvp_register_c(char *key, char *dest){
	kvp_register(key,kvp_store_c,dest);
}

void kvp_register_uc(char *key, unsigned char *dest){
	kvp_register(key,kvp_store_uc,dest);
}

/*Register short*/
void kvp_register_s(char *key, short *dest){
	kvp_register(key,kvp_store_s,dest);
}

void kvp_register_us(char *key, unsigned short *dest){
	kvp_register(key,kvp_store_us,dest);
}

/*Register int*/
void kvp_register_i(char *key, int *dest){
	kvp_register(key,kvp_store_i,dest);
}

void kvp_register_ui(char *key, unsigned int *dest){
	kvp_register(key,kvp_store_ui,dest);
}

/*Register long*/
void kvp_register_l(char *key, long *dest){
	kvp_register(key,kvp_store_l,dest);
}

void kvp_register_ul(char *key, unsigned long *dest){
	kvp_register(key,kvp_store_ul,dest);
}

/*Register long long*/
void kvp_register_ll(char *key, long long *dest){
	kvp_register(key,kvp_store_ll,dest);
}

void kvp_register_ull(char *key, unsigned long long *dest){
	kvp_register(key,kvp_store_ull,dest);
}

/*Register float*/
void kvp_register_f(char *key, float *dest){
	kvp_register(key,kvp_store_f,dest);
}

/*Register double*/
void kvp_register_d(char *key, double *dest){
	kvp_register(key,kvp_store_d,dest);
}

/*Register string (char*)*/
void kvp_register_string(char *key, char *dest, size_t maxlen){
	kvp_held_string *khs = (kvp_held_string*)malloc(sizeof(kvp_held_string));
	khs->string = dest;
	khs->size = maxlen;
	kvp_register_core(key, kvp_store_string, khs, 1);
}

/*End convenience functions to register storing to variables*/

/* Handle a given found key value pair*/
KVP_ERROR_TYPE kvp_handle(char *key, char *value){
	int i;
	for (i=0;i<kvp_itemcount;++i){
		if (!strcmp(key,kvp_items[i].name)){
			return kvp_items[i].action(key,value,kvp_items[i].data);
		}
	}
	return KVP_BAD_KEY;
}

/*Function to act on raw character text*/
void kvp_from_text(char *text){
	unsigned char comment=0, lock=0, inquotes=0;

	char **part;
	char *keystart=NULL, *valstart=NULL;
	int len, linect;

	keystart = text;
	part = &keystart;
	len = strlen(text);
	linect = 1;
	int i;

	for (i = 0;i<=len;++i){
		/*Toggle comment on*/
		if (text[i]=='#') comment = 1;

		/*Deal with end of line*/
		if (text[i]=='\n' || i == len){
			text[i]='\0';
			if (strlen(keystart)!=0){
				KVP_ERROR_TYPE val = kvp_handle(keystart, valstart);
				if (val != KVP_OK){
					printf("Unable to parse line %i\n", linect);
					if (val & KVP_BAD_KEY) printf("Key %s is not found in list of known keys\n",keystart);
					if (val & KVP_BAD_VALUE) printf("Unable to parse %s as specified type\n", valstart);
					if (val & KVP_BAD_RANGE) printf("Value %s out of range for specified type\n", valstart);
					exit(-1);
				}
			}
			keystart = text + i + 1;
			valstart = NULL;
			part = &keystart;
			comment = 0;
			lock = 0;
			linect++;
			continue;
		}

		/*Replace white space or comments with NULL*/
		if ((KVP_ISWHITESPACE(text[i]) && !inquotes) || comment){
			text[i]='\0';
			/*Trim off leading spaces only*/
			if (!lock) (*part)++;
			continue;
		}

		/*Note that there is no escaping of quotes - you cannot have quotes
		 *inside strings*/
		if (KVP_ISQUOTE(text[i])) {
			inquotes = !inquotes;
			text[i]='\0';
			if (!lock) (*part)++;
			continue;
		}

		/*Now NOT white space and not quote start*/
		lock = 1;

		/*Found separator*/
		if (KVP_ISSEPARATOR(text[i])){
			text[i]='\0';
			valstart = text + i + 1;
			part = &valstart;
			lock = 0;
			continue;
		}
	}
}

/*Copy input string so that it is unchanged*/

void kvp_from_const_text(const char* text){
	char *copy;
	copy = (char*)malloc(sizeof(char) * (strlen(text)+1));
	strcpy(copy, text);
	kvp_from_text(copy);
	free(copy);
}

/*Load files*/
void kvp_from_file(const char* filename){
	FILE *fptr;
	fptr = fopen(filename, "r");
	if (!fptr) {
		printf("Unable to open file %s\n",filename);
		exit(-1);
	}
	fseek(fptr,0,SEEK_END);
	long sz = ftell(fptr);
	char *text = (char*)malloc(sizeof(char) * sz);
	rewind(fptr);
	fread(text,sizeof(char),sz,fptr);
	kvp_from_text(text);
	free(text);
	fclose(fptr);
}

/*Deallocate internal memory to the KVP system*/
void kvp_free(){

	if (kvp_items) {
		int i;
		/*If we are holding the only copy of an allocated pointer deallocate it*/
		for (i = 0; i < kvp_itemcount; ++i){
			if (kvp_items[i].free) free(kvp_items[i].data);
		}
		free(kvp_items);
	}
	kvp_items = NULL;
	kvp_itemcount=0;
	kvp_itemslots=0;
}
