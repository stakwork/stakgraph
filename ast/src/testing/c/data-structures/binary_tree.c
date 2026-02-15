#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node *left;
    struct Node *right;
} TreeNode;

TreeNode* tree_insert(TreeNode *root, int value) {
    if (!root) {
        TreeNode *n = malloc(sizeof(TreeNode));
        n->value = value;
        n->left = n->right = NULL;
        return n;
    }
    
    if (value < root->value) {
        root->left = tree_insert(root->left, value);
    } else {
        root->right = tree_insert(root->right, value);
    }
    return root;
}

void tree_free(TreeNode *root) {
    if (root) {
        tree_free(root->left);
        tree_free(root->right);
        free(root);
    }
}
