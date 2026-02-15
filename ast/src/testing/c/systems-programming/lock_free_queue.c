#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct Node {
    void *data;
    _Atomic(struct Node*) next;
} Node;

typedef struct {
    _Atomic(Node*) head;
    _Atomic(Node*) tail;
} LockFreeQueue;

LockFreeQueue* queue_create(void) {
    LockFreeQueue *q = malloc(sizeof(LockFreeQueue));
    Node *dummy = malloc(sizeof(Node));
    dummy->next = NULL;
    atomic_init(&q->head, dummy);
    atomic_init(&q->tail, dummy);
    return q;
}

void queue_enqueue(LockFreeQueue *q, void *data) {
    Node *new_node = malloc(sizeof(Node));
    new_node->data = data;
    atomic_init(&new_node->next, NULL);
    
    Node *tail;
    while (1) {
        tail = atomic_load(&q->tail);
        Node *next = atomic_load(&tail->next);
        
        if (tail == atomic_load(&q->tail)) {
            if (next == NULL) {
                if (atomic_compare_exchange_weak(&tail->next, &next, new_node)) {
                    break;
                }
            } else {
                atomic_compare_exchange_weak(&q->tail, &tail, next);
            }
        }
    }
    atomic_compare_exchange_weak(&q->tail, &tail, new_node);
}
