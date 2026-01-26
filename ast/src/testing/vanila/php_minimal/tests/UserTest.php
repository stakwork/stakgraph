<?php

require_once __DIR__ . '/../models/User.php';

// Mocking assert for standalone execution
function assert_true($condition, $message = "Assertion failed") {
    if (!$condition) {
        echo "FAIL: $message\n";
        exit(1);
    }
    echo "PASS: $message\n";
}

$user = new User('testuser', 'test@example.com');
assert_true($user->username === 'testuser', "Username set correctly");
$id = $user->save();
assert_true(!empty($id), "User saved with ID");

$found = User::find($id);
assert_true($found->email === 'test@example.com', "User retrieved correctly");
