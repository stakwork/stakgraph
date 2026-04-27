#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// @ast node: Class "Entry"
// @ast node: Class "Entry"
typedef struct Entry {
    char *key;
    void *value;
    struct Entry *next;
} Entry;

// @ast node: Class "HashTable"
typedef struct {
    Entry **buckets;
    size_t size;
} HashTable;

// @ast node: Function "hash"
static uint32_t hash(const char *key) {
    uint32_t h = 0x811c9dc5;
    while (*key) {
        h ^= (uint8_t)*key++;
        h *= 0x01000193;
    }
    return h;
}

// @ast node: Function "ht_create"
HashTable* ht_create(size_t size) {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->size = size;
    ht->buckets = calloc(size, sizeof(Entry*));
    return ht;
}

// @ast node: Function "ht_put"
// @ast edge: Calls -> Function "hash" "hash_table.c"
void ht_put(HashTable *ht, const char *key, void *value) {
    uint32_t idx = hash(key) % ht->size;
    Entry *e = ht->buckets[idx];
    
    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return;
        }
        e = e->next;
    }
    
    Entry *new_entry = malloc(sizeof(Entry));
    new_entry->key = strdup(key);
    new_entry->value = value;
    new_entry->next = ht->buckets[idx];
    ht->buckets[idx] = new_entry;
}

// @ast node: Function "ht_get"
// @ast edge: Calls -> Function "hash" "hash_table.c"
void* ht_get(HashTable *ht, const char *key) {
    uint32_t idx = hash(key) % ht->size;
    Entry *e = ht->buckets[idx];
    
    while (e) {
        if (strcmp(e->key, key) == 0) {
            return e->value;
        }
        e = e->next;
    }
    return NULL;
}
