<?php

require_once __DIR__ . '/../models/User.php';
require_once __DIR__ . '/../includes/helpers.php';

class UserController {
    public function index() {
        $users = User::all();
        jsonResponse($users);
    }

    public function show($id) {
        $user = User::find($id);
        if ($user) {
            jsonResponse($user);
        } else {
            jsonResponse(['error' => 'User not found'], 404);
        }
    }

    public function create() {
        $data = json_decode(file_get_contents('php://input'), true);
        if (!isset($data['username']) || !isset($data['email'])) {
            jsonResponse(['error' => 'Invalid input'], 400);
        }

        $user = new User(sanitize($data['username']), sanitize($data['email']));
        $user->save();
        
        jsonResponse(['id' => $user->id, 'message' => 'User created'], 201);
    }
}
