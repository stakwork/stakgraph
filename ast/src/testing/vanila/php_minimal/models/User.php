<?php

require_once __DIR__ . '/../database.php';

class User {
    public $id;
    public $username;
    public $email;

    public function __construct($username, $email) {
        $this->username = $username;
        $this->email = $email;
    }

    public function save() {
        $db = Database::getInstance();
        $this->id = $db->insert('users', [
            'username' => $this->username,
            'email' => $this->email
        ]);
        return $this->id;
    }

    public static function find($id) {
        $db = Database::getInstance();
        $data = $db->find('users', $id);
        if ($data) {
            $user = new User($data['username'], $data['email']);
            $user->id = $data['id'];
            return $user;
        }
        return null;
    }

    public static function all() {
        $db = Database::getInstance();
        return $db->findAll('users');
    }
}
