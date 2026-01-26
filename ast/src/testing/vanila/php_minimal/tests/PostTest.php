<?php

require_once __DIR__ . '/../models/Post.php';

function assert_equals($expected, $actual, $message = "") {
    if ($expected !== $actual) {
        echo "FAIL: $message - Expected $expected, got $actual\n";
        exit(1);
    }
    echo "PASS: $message\n";
}

$post = new Post('My Title', 'Content here', 'user1');
$id = $post->save();

assert_equals('My Title', $post->title, "Title matches");
assert_equals('user1', $post->user_id, "User ID matches");
